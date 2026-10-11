#!/usr/bin/env python3
"""TensorRT export + inference for AASIST. Self-contained (no shared package).

Exports only the model's `net` (all preprocessing already lives in the original
`score_batch`) with a fixed time axis and a dynamic batch axis, builds a FP16
engine (FP32 fallback if parity drifts), finds the fastest batch on the current
GPU, and exposes a drop-in `AASISTTRT` class identical to the PyTorch path except
the neural forward runs on TensorRT.

CLI:
  python trt_aasist.py export   # ONNX -> engine -> parity -> sweep -> sidecar
  python trt_aasist.py sweep     # re-run the batch sweep, update sidecar
  python trt_aasist.py parity    # PyTorch vs TRT parity report
  python trt_aasist.py score AUDIO.wav

Pin the GPU with:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>
"""
from __future__ import annotations
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # import dir-local entry + _net
# Pin GPU deterministically: PCI order makes CUDA indices match `nvidia-smi`.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("SSB_TRT_GPU", "3"))

import torch  # noqa: E402  (after env pin)

# ======================= per-model config =======================
ENTRY_MODULE   = "aasist"          # module exposing the AntiSpoofingModel subclass
ENTRY_CLASS    = "AASIST"          # the subclass name
SLUG           = "aasist"
PARITY_DATASET = "InTheWild"       # sibling dataset dir with data/*.parquet
MAX_BATCH_CAP  = 32                # VRAM ceiling for the profile + sweep
PARITY_CHUNK   = 8                 # safe mini-batch for the parity comparison
OPSET          = 17
# Keep FP16 iff it preserves the score RANKING (Spearman) -> identical EER.
# This is the metric that matters for the benchmark and is scale-invariant, so
# small absolute-logit drift (harmless for EER) does not force an FP32 fallback.
# FP16 is also mandatory for the largest models (FP32 would not fit in VRAM).
PARITY_SPEARMAN_TOL = 0.9999       # min Spearman rank-corr to keep FP16
PARITY_FLOOR        = 0.99         # hard floor: below this the engine is wrong -> FAIL
PARITY_MAD_TOL = 1e-2              # informational only
PARITY_R_TOL   = 0.9999           # informational only
FORCE_FP32         = False
FORCE_FP16         = False   # skip FP32 (for giant models where FP32 won't fit VRAM)
DYNAMO_EXPORT      = False   # use the dynamo exporter + external data (models >2GB)
ALLOW_ORT_FALLBACK = False
# ================================================================

from importlib import import_module as _imp  # noqa: E402
_OrigClass = getattr(_imp(ENTRY_MODULE), ENTRY_CLASS)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def gpu_slug() -> str:
    name = torch.cuda.get_device_name(0)
    return name.replace("NVIDIA ", "").replace("GeForce ", "").strip().replace(" ", "_")


def load_model():
    m = _OrigClass()
    m.load()
    return m


def real_audio(n=64):
    """Decode up to n real 16 kHz mono utterances from PARITY_DATASET/data/*.parquet."""
    import pyarrow.parquet as pq
    import soundfile as sf
    import torchaudio.functional as AF

    data_dir = HERE.parent / PARITY_DATASET / "data"
    files = sorted(data_dir.glob("test-*.parquet")) or sorted(data_dir.glob("*.parquet"))
    out = []
    for f in files:
        t = pq.read_table(f)
        col = "audio" if "audio" in t.column_names else t.column_names[0]
        for row in t.column(col).to_pylist():
            b = row["bytes"] if isinstance(row, dict) else row
            if not b:
                continue
            a, sr = sf.read(io.BytesIO(b), dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            a = np.ascontiguousarray(a, dtype=np.float32)
            if sr != 16000:
                a = AF.resample(torch.from_numpy(a), sr, 16000).numpy().astype(np.float32)
                sr = 16000
            out.append((a, sr))
            if len(out) >= n:
                return out
    if not out:
        raise RuntimeError(f"no parity audio found under {data_dir}")
    return out


class _Capture:
    """Wrap net: pass through to the real net, record input tensor + output."""

    def __init__(self, net):
        self.net = net
        self.x = None
        self.out = None

    def __call__(self, x, *a, **k):
        self.x = x.detach()
        self.out = self.net(x, *a, **k)
        return self.out

    def __getattr__(self, name):
        return getattr(self.net, name)


def _logits_index(out):
    """Return (L, i, n_classes): tuple length (None if tensor), logits slot, n_classes.

    Heuristic: the class-logits tensor is the 2-D (B, C) tensor with the smallest C.
    """
    if isinstance(out, torch.Tensor):
        return None, None, int(out.shape[-1])
    cands = [(j, t) for j, t in enumerate(out)
             if isinstance(t, torch.Tensor) and t.dim() == 2]
    if not cands:
        raise RuntimeError("could not locate a 2-D (B,C) logits tensor in net output")
    j, t = min(cands, key=lambda it: int(it[1].shape[-1]))
    return len(out), j, int(t.shape[-1])


def analyze(model):
    """One real forward through the capture shim -> (T, L, i, n_classes)."""
    data = real_audio(1)
    audios = [a for a, _ in data]
    srs = [s for _, s in data]
    cap = _Capture(model.net)
    model.net = cap
    with torch.no_grad():
        model.score_batch(audios, srs)
    model.net = cap.net
    T = int(cap.x.shape[-1])
    L, i, n_classes = _logits_index(cap.out)
    return T, L, i, n_classes


def _extractor(L, i):
    """Pick the logits tensor out of a net's raw output."""
    if L is None:
        return lambda y: y
    return lambda y, i=i: y[i]


def _rebuild(L, i):
    """Wrap a bare logits tensor back into the net's original output structure."""
    if L is None:
        return lambda y: y
    return lambda y, L=L, i=i: tuple(y if j == i else None for j in range(L))


def _prep_for_export(net):
    """Make export-hostile layers traceable. No-op for non-fairseq models.

    fairseq wav2vec2/hubert call `pad_to_multiple`, which does `(tsz/multiple)
    .is_integer()`; under torch.jit tracing `tsz` becomes a Tensor with no
    `.is_integer()`. Our time axis is static, so we swap in a constant-length
    pad that traces cleanly. Patches every fairseq module that bound the name.
    """
    def _safe_pad(x, multiple, dim=-1, value=0):
        import torch.nn.functional as F
        if x is None:
            return None, 0
        tsz = int(x.shape[dim])                      # static: time axis is fixed
        rem = (multiple - tsz % multiple) % multiple
        if rem == 0:
            return x, 0
        pad_offset = (0,) * (-1 - dim) * 2
        return F.pad(x, (*pad_offset, 0, rem), value=value), rem

    for modname in ("fairseq.models.wav2vec.utils",
                    "fairseq.models.wav2vec.wav2vec2",
                    "fairseq.models.hubert.hubert"):
        mod = sys.modules.get(modname)
        if mod is not None and hasattr(mod, "pad_to_multiple"):
            mod.pad_to_multiple = _safe_pad
    _freeze_sinc(net)
    # optional per-model export patch (dir-local `_trt_patch.py` with `patch(net)`)
    try:
        import importlib
        importlib.import_module("_trt_patch").patch(net)
    except ModuleNotFoundError:
        pass
    if DYNAMO_EXPORT:
        _replace_global_avgpool(net)
    return net


class _MeanPool(torch.nn.Module):
    """Global average over `dims` (keepdim) — == AdaptiveAvgPool{1,2}d(1)."""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.mean(dim=self.dims, keepdim=True)


def _replace_global_avgpool(net):
    """Swap AdaptiveAvgPool1d/2d(output_size=1) for an explicit mean. The dynamo
    exporter lowers the adaptive pool to as_strided/SequenceEmpty, which TensorRT
    rejects; a plain mean lowers to ReduceMean. Identical for output_size==1."""
    import torch.nn as nn
    for full_name, mod in list(net.named_modules()):
        is1d = isinstance(mod, nn.AdaptiveAvgPool1d) and mod.output_size in (1, (1,))
        is2d = isinstance(mod, nn.AdaptiveAvgPool2d) and mod.output_size in (1, (1, 1))
        if not (is1d or is2d):
            continue
        parent = net
        *parents, attr = full_name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, attr, _MeanPool((-1,) if is1d else (-2, -1)))
    return net


def _freeze_sinc(net):
    """Replace SincConv-style layers with an equivalent nn.Conv1d holding the
    precomputed band-pass filters. At eval the filters are constant, but their
    in-forward construction (torch.sin/cat/flip from learnable params) either
    won't build in TensorRT or constant-folds to wrong values. Baking them into a
    plain Conv1d removes the sinc math from the graph. No-op when no Sinc layer.
    """
    import torch.nn as nn
    sincs = [(n, m) for n, m in net.named_modules() if "Sinc" in type(m).__name__]
    if not sincs:
        return net
    dev = next(net.parameters()).device
    for full_name, mod in sincs:
        kernel = int(getattr(mod, "kernel_size", 0)) or 1
        with torch.no_grad():
            try:
                mod(torch.zeros(1, 1, max(kernel * 4, 4096), device=dev))
            except Exception:  # noqa: BLE001 — filters are set before the conv call
                pass
        W = mod.filters.detach().clone()          # [out, 1, kernel] (or [out, kernel])
        if W.dim() == 2:
            W = W.unsqueeze(1)
        conv = nn.Conv1d(W.shape[1], W.shape[0], W.shape[2],
                         stride=int(getattr(mod, "stride", 1)),
                         padding=int(getattr(mod, "padding", 0)),
                         dilation=int(getattr(mod, "dilation", 1)),
                         bias=False).to(dev).eval()
        conv.weight.data.copy_(W)
        parent = net
        *parents, attr = full_name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, attr, conv)
    return net


class _ExportNet(torch.nn.Module):
    """forward(x[B,T]) -> logits[B,C] (single tensor) for ONNX/TRT."""

    def __init__(self, net, L, i):
        super().__init__()
        self.net = net
        self._extract = _extractor(L, i)

    def forward(self, x):
        return self._extract(self.net(x))


# ----------------------------------------------------------------------------
# export + build
# ----------------------------------------------------------------------------
def export_onnx(model, T, L, i, onnx_path):
    net = _prep_for_export(model.net)
    wrap = _ExportNet(net, L, i).eval().to("cuda")
    dummy = torch.zeros(2, T, device="cuda", dtype=torch.float32)
    if DYNAMO_EXPORT:
        # >2 GB models: TorchScript exporter's shape-inference overflows the 2 GB
        # protobuf limit. The dynamo exporter writes weights as external data.
        batch = torch.export.Dim("b", min=1, max=MAX_BATCH_CAP)
        torch.onnx.export(
            wrap, (dummy,), str(onnx_path), dynamo=True, external_data=True,
            input_names=["wav"], output_names=["logits"],
            dynamic_shapes={"x": {0: batch}},
        )
    else:
        torch.onnx.export(
            wrap, dummy, str(onnx_path), opset_version=OPSET,
            input_names=["wav"], output_names=["logits"],
            dynamic_axes={"wav": {0: "batch"}, "logits": {0: "batch"}},
            do_constant_folding=True,
        )
    return onnx_path


def build_engine(onnx_path, T, precision, max_batch, opt_batch, engine_path, timing_cache):
    import tensorrt as trt

    sev = trt.Logger.VERBOSE if os.environ.get("SSB_TRT_VERBOSE") else trt.Logger.WARNING
    logger = trt.Logger(sev)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file resolves external-data sidecars (needed for >2 GB models);
    # works for inline ONNX too.
    if not parser.parse_from_file(str(onnx_path)):
        errs = "; ".join(str(parser.get_error(k)) for k in range(parser.num_errors))
        raise RuntimeError(f"onnx parse failed: {errs}")

    cfg = builder.create_builder_config()
    cfg.builder_optimization_level = 1                       # minimum build time
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)

    tc_bytes = Path(timing_cache).read_bytes() if Path(timing_cache).exists() else b""
    tc = cfg.create_timing_cache(tc_bytes)
    cfg.set_timing_cache(tc, ignore_mismatch=False)

    profile = builder.create_optimization_profile()
    profile.set_shape("wav", (1, T), (opt_batch, T), (max_batch, T))
    cfg.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError("engine build returned None")
    Path(engine_path).write_bytes(bytes(plan))
    Path(timing_cache).write_bytes(bytes(tc.serialize()))
    return engine_path


# ----------------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------------
class _TRTCallable:
    """Mimics net(xt): runs the engine on a [B,T] float32 CUDA tensor."""

    def __init__(self, engine_path, n_classes, L, i):
        import tensorrt as trt

        self.n_classes = n_classes
        self.rebuild = _rebuild(L, i)
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError(
                "could not create execution context (likely OOM reserving max-profile "
                "memory) — lower MAX_BATCH_CAP")
        # resolve I/O tensor names
        self.in_name, self.out_name = "wav", "logits"

    def __call__(self, x, *a, **k):
        x = x.to("cuda", torch.float32).contiguous()
        B = x.shape[0]
        self.ctx.set_input_shape(self.in_name, tuple(x.shape))
        out = torch.empty((B, self.n_classes), device="cuda", dtype=torch.float32)
        self.ctx.set_tensor_address(self.in_name, x.data_ptr())
        self.ctx.set_tensor_address(self.out_name, out.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        self.ctx.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()
        return self.rebuild(out)


# ----------------------------------------------------------------------------
# parity + sweep
# ----------------------------------------------------------------------------
def _chunked_scores(model, audios, srs, chunk):
    out = []
    for k in range(0, len(audios), chunk):
        out.extend(model.score_batch(audios[k:k + chunk], srs[k:k + chunk]))
    return np.asarray(out, dtype=np.float64)


def _spearman(a, b):
    if len(a) < 2:
        return 1.0
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def parity(model, trt_call, n=64, chunk=PARITY_CHUNK):
    data = real_audio(n)
    audios = [a for a, _ in data]
    srs = [s for _, s in data]
    torch_net = model.net
    py = _chunked_scores(model, audios, srs, chunk)
    model.net = trt_call
    tr = _chunked_scores(model, audios, srs, chunk)
    model.net = torch_net
    mad = float(np.max(np.abs(py - tr)))
    pear = float(np.corrcoef(py, tr)[0, 1]) if len(py) > 1 else 1.0
    spear = _spearman(py, tr)
    return {"n": len(py), "max_abs_score_diff": mad, "pearson": pear,
            "spearman": spear}


def sweep(model, trt_call,
          batches=(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128), iters=20):
    a, sr = real_audio(1)[0]
    model.net = trt_call
    res = {}
    for B in batches:
        if B > MAX_BATCH_CAP:
            break
        ab, sb = [a] * B, [sr] * B
        try:
            for _ in range(3):
                model.score_batch(ab, sb)                    # warmup
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                model.score_batch(ab, sb)
            torch.cuda.synchronize()
            dt = time.time() - t0
            res[B] = B * iters / dt
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                break
            raise
    best = max(res, key=res.get)
    return best, res


# ----------------------------------------------------------------------------
# drop-in inference class
# ----------------------------------------------------------------------------
class AASISTTRT(_OrigClass):
    """Drop-in: original preprocessing/score_batch; net replaced by the TRT engine."""

    def load(self):
        self.device = "cuda"
        side = json.loads((HERE / f"trt_{SLUG}.json").read_text())[gpu_slug()]
        eng = HERE / side["engine"]
        self.net = _TRTCallable(str(eng), side["n_classes"], side["L"], side["i"])
        self.batch_size = side["best_batch"]


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _do_export():
    gpu = gpu_slug()
    side_path = HERE / f"trt_{SLUG}.json"
    tc = HERE / f".trt_timing_{gpu}.cache"
    m = load_model()
    T, L, i, n_classes = analyze(m)
    print(f"[analyze] T={T} n_classes={n_classes} L={L} i={i}")
    onnx_path = HERE / f"{SLUG}.onnx"
    export_onnx(m, T, L, i, onnx_path)
    print(f"[onnx] wrote {onnx_path.name}")

    # PyTorch reference scores while the model is on GPU, then free it so the
    # engine build + TRT inference never co-reside with the model (giant >2 GB
    # models would otherwise OOM the 16 GB card).
    pdata = real_audio(64)
    paud, psr = [a for a, _ in pdata], [s for _, s in pdata]
    py = _chunked_scores(m, paud, psr, PARITY_CHUNK)
    m.net.to("cpu")
    torch.cuda.empty_cache()

    opt_batch = min(32, MAX_BATCH_CAP)
    if FORCE_FP16:
        precisions = ["fp16"]
    elif FORCE_FP32:
        precisions = ["fp32"]
    else:
        precisions = ["fp16", "fp32"]
    chosen = None
    last_err = None
    for prec in precisions:
        eng = HERE / f"engine_{gpu}_{prec}_b1-{opt_batch}-{MAX_BATCH_CAP}.plan"
        try:
            t0 = time.time()
            build_engine(str(onnx_path), T, prec, MAX_BATCH_CAP, opt_batch, str(eng), str(tc))
            bt = time.time() - t0
            trt_call = _TRTCallable(str(eng), n_classes, L, i)
            m.net = trt_call
            tr = _chunked_scores(m, paud, psr, PARITY_CHUNK)
            p = {"n": len(py),
                 "max_abs_score_diff": float(np.max(np.abs(py - tr))),
                 "pearson": float(np.corrcoef(py, tr)[0, 1]) if len(py) > 1 else 1.0,
                 "spearman": _spearman(py, tr)}
        except Exception as e:  # noqa: BLE001 — try the next precision (e.g. FP16 layer not buildable)
            last_err = e
            print(f"[{prec}] FAILED: {type(e).__name__}: {e}")
            continue
        print(f"[{prec}] build={bt:.1f}s parity={p}")
        chosen = (prec, eng, p, trt_call)
        if prec == "fp16" and p["spearman"] >= PARITY_SPEARMAN_TOL:
            break

    if chosen is None:
        raise RuntimeError(f"all precisions failed to build; last error: {last_err}")
    prec, eng, p, trt_call = chosen
    if p["spearman"] < PARITY_FLOOR:
        raise RuntimeError(
            f"parity too low (spearman={p['spearman']:.4f} < {PARITY_FLOOR}): "
            f"engine output does not match PyTorch — not accepting")
    m.net = trt_call
    best, table = sweep(m, trt_call)
    side = json.loads(side_path.read_text()) if side_path.exists() else {}
    side[gpu] = {
        "precision": prec, "engine": eng.name, "window_samples": T,
        "n_classes": n_classes, "L": L, "i": i, "best_batch": best,
        "throughput_utt_s": {str(k): round(v, 2) for k, v in table.items()},
        "parity": p, "trt_version": __import__("tensorrt").__version__,
    }
    side_path.write_text(json.dumps(side, indent=2, default=str))
    print(f"[done] {SLUG}: prec={prec} best_batch={best} "
          f"utt/s={table[best]:.1f} parity_mad={p['max_abs_score_diff']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "sweep", "parity", "score"])
    ap.add_argument("audio", nargs="?")
    args = ap.parse_args()
    gpu = gpu_slug()
    side_path = HERE / f"trt_{SLUG}.json"

    if args.cmd == "export":
        _do_export()
    elif args.cmd in ("sweep", "parity"):
        m = load_model()
        side = json.loads(side_path.read_text())[gpu]
        eng = HERE / side["engine"]
        trt_call = _TRTCallable(str(eng), side["n_classes"], side["L"], side["i"])
        if args.cmd == "parity":
            print(parity(m, trt_call))
        else:
            best, table = sweep(m, trt_call)
            full = json.loads(side_path.read_text())
            full[gpu]["best_batch"] = best
            full[gpu]["throughput_utt_s"] = {str(k): round(v, 2) for k, v in table.items()}
            side_path.write_text(json.dumps(full, indent=2, default=str))
            print(f"best_batch={best} utt/s={table[best]:.1f}")
    elif args.cmd == "score":
        import soundfile as sf
        a, sr = sf.read(args.audio, dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        m = AASISTTRT()
        m.load()
        print(m.score_batch([a.astype(np.float32)], [sr])[0])


if __name__ == "__main__":
    main()
