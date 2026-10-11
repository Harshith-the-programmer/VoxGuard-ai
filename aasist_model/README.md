---
license: mit
tags:
  - audio
  - anti-spoofing
  - audio-deepfake-detection
  - speech
  - asvspoof
---

# AASIST

[![EER% 0.83 on ASVspoof2019_LA](https://img.shields.io/badge/EER%25%20on%20ASVspoof2019__LA-0.83%25-brightgreen)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 12.35 on ASVspoof2021_LA](https://img.shields.io/badge/EER%25%20on%20ASVspoof2021__LA-12.35%25-yellow)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 17.04 on ASVspoof2021_DF](https://img.shields.io/badge/EER%25%20on%20ASVspoof2021__DF-17.04%25-yellow)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 43.01 on InTheWild](https://img.shields.io/badge/EER%25%20on%20InTheWild-43.01%25-red)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 51.05 on CD-ADD](https://img.shields.io/badge/EER%25%20on%20CD--ADD-51.05%25-red)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 57.61 on SONAR](https://img.shields.io/badge/EER%25%20on%20SONAR-57.61%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 37.98 on LibriSeVoc](https://img.shields.io/badge/EER%25%20on%20LibriSeVoc-37.98%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 46.11 on CFAD](https://img.shields.io/badge/EER%25%20on%20CFAD-46.11%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 54.35 on CVoiceFake_small](https://img.shields.io/badge/EER%25%20on%20CVoiceFake__small-54.35%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 35.53 on ASVspoof5](https://img.shields.io/badge/EER%25%20on%20ASVspoof5-35.53%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 46.02 on DeepVoice](https://img.shields.io/badge/EER%25%20on%20DeepVoice-46.02%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 46.07 on ArAD](https://img.shields.io/badge/EER%25%20on%20ArAD-46.07%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 29.74 on DECRO](https://img.shields.io/badge/EER%25%20on%20DECRO-29.74%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 39.82 on J-SPAW_LA](https://img.shields.io/badge/EER%25%20on%20J--SPAW__LA-39.82%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 50.41 on ODSS](https://img.shields.io/badge/EER%25%20on%20ODSS-50.41%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 37.36 on HABLA](https://img.shields.io/badge/EER%25%20on%20HABLA-37.36%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 39 on DFADD](https://img.shields.io/badge/EER%25%20on%20DFADD-39%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 29.89 on PyAra](https://img.shields.io/badge/EER%25%20on%20PyAra-29.89%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 47.11 on XMAD](https://img.shields.io/badge/EER%25%20on%20XMAD-47.11%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![1-SRR% 30.83 on LRLspoof](https://img.shields.io/badge/1--SRR%25%20on%20LRLspoof-30.83%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 30.93 on ADD22_eval_31](https://img.shields.io/badge/EER%25%20on%20ADD22__eval__31-30.93%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 47.75 on ADD2023_track12_test_r1](https://img.shields.io/badge/EER%25%20on%20ADD2023__track12__test__r1-47.75%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![EER% 13.6 on EmoFake_test](https://img.shields.io/badge/EER%25%20on%20EmoFake__test-13.6%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![1-SRR% 78.54 on EmoSpoofTTS](https://img.shields.io/badge/1--SRR%25%20on%20EmoSpoofTTS-78.54%25-lightgrey)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![arena tier](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/aasist/tier.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)
[![arena rank](https://img.shields.io/endpoint?url=https://speechantispoofingbenchmarks-speechantispoofingarena.hf.space/badge/aasist/rank.json)](https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena?system=aasist)

AASIST audio anti-spoofing (voice-deepfake detection) countermeasure from
*"AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention
Networks"* (Jung et al., ICASSP 2022). This is the **official `AASIST` variant**
(not AASIST-L), using the upstream [clovaai/aasist](https://github.com/clovaai/aasist)
ASVspoof2019 LA pretrained checkpoint. The model takes a raw speech waveform and
returns a score where **higher = more bona fide**.

- **Code:** https://github.com/clovaai/aasist
- **Paper:** https://arxiv.org/abs/2110.01200
- **Parameters:** 297,866 (0.298 M)
- **Checkpoint:** [`AASIST.pth`](./AASIST.pth)

This repo is self-contained for inference: the network definition is in
[`_net.py`](./_net.py) and the exact wrapper used to produce the Arena scores in
[`aasist.py`](./aasist.py).

## Architecture

AASIST operates directly on the raw waveform: a sinc-convolution front-end and a
RawNet2-style residual encoder produce a spectro-temporal feature map, which is
modelled by heterogeneous stacking graph attention layers over spectral and
temporal sub-graphs with a learnable max/average readout, followed by a 2-class
output (bona fide vs. spoof). The Arena score is the bona-fide logit.

## Reproducing the Arena scores

Inference uses a deterministic first-64600-sample window (no random crop),
matching the upstream `data_utils.pad()` used at eval. Audio is provided as
float32 mono at 16 kHz (no resampling in the wrapper).

```python
from aasist import AASIST
m = AASIST(); m.load()
scores = m.score_batch([wav], [16000])   # higher = more bona fide
```

| Dataset | EER % | n_trials |
|---------|------:|---------:|
| ASVspoof2019_LA (in-domain) | 0.83 | 71,237 |
| ASVspoof2021_LA | 12.35 | 181,566 |
| ASVspoof2021_DF | 17.04 | 611,829 |
| InTheWild | 43.01 | 31,779 |
| CD-ADD | 51.05 | 20,786 |

The in-domain ASVspoof2019 LA result reproduces the paper's reported EER (~0.83%).

## License

MIT (inherited from clovaai/aasist; see [`LICENSE`](./LICENSE)).

## Maintainer

Maintained by Kirill Borodin (SpeechAntiSpoofingBenchmarks).
- Email: kborodin.research@gmail.com
- Telegram: [@korallll_ai](https://t.me/korallll_ai)
