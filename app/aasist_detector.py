import json
import math
import sys
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parent.parent
AASIST_DIR = BASE_DIR / "aasist_source"
THRESHOLD_PATH = BASE_DIR / "aasist_threshold.json"

if str(AASIST_DIR) not in sys.path:
    sys.path.insert(0, str(AASIST_DIR))

from models.AASIST import Model


DEVICE = torch.device("cpu")
AUDIO_LENGTH = 64600
SAMPLE_RATE = 16000
torch.set_num_threads(1)

@lru_cache(maxsize=1)
def load_aasist_model():
    """Load the AASIST model and its trained checkpoint once."""
    config_path = AASIST_DIR / "config" / "AASIST.conf"
    checkpoint_path = AASIST_DIR / "models" / "weights" / "AASIST.pth"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"AASIST configuration not found: {config_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"AASIST checkpoint not found: {checkpoint_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    model = Model(config["model_config"]).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print("AASIST model loaded successfully!")
    return model


@lru_cache(maxsize=1)
def load_classification_threshold():
    """Load and validate the experimental classification threshold."""
    if not THRESHOLD_PATH.is_file():
        print("Threshold file not found.")
        return None

    try:
        with THRESHOLD_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        threshold = float(data["threshold"])

        if not math.isfinite(threshold):
            return None

        if data.get("positive_class") != "bonafide":
            return None

        if data.get("higher_score_means") != "more_bona_fide_like":
            return None

        if data.get("status") != "EXPERIMENTAL_INTERNAL_HOLDOUT":
            return None

        return data

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        print("Threshold file is invalid.")
        return None


def decode_audio_ffmpeg(file_path: str) -> np.ndarray:
    """Decode audio to mono float32 at 16 kHz using FFmpeg."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(file_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "FFmpeg is unavailable in the AI service PATH."
        ) from error
    except subprocess.CalledProcessError as error:
        details = error.stderr.decode(
            "utf-8", errors="replace"
        ).strip()

        raise ValueError(
            "FFmpeg could not decode the recording: " + details[-500:]
        ) from error

    if not completed.stdout:
        raise ValueError("No decodable audio was found.")

    audio = np.frombuffer(
        completed.stdout,
        dtype="<f4",
    )

    if audio.size == 0 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio is empty or contains invalid samples.")

    return audio


def prepare_audio_segments(file_path: str):
    """
    Prepare fixed-length audio windows for AASIST.

    Audio shorter than one window is repeated to fill the required length.
    Longer recordings use complete windows. If a remainder exists, an
    end-aligned window is added rather than padding a tiny remainder
    into a separate full-length segment.
    """
    audio = decode_audio_ffmpeg(file_path)

    if audio.size == 0:
        raise ValueError("Audio is empty.")

    # Short recordings require one fixed-length model input.
    if len(audio) <= AUDIO_LENGTH:
        if len(audio) < AUDIO_LENGTH:
            repeats = math.ceil(AUDIO_LENGTH / len(audio))
            audio = np.tile(audio, repeats)[:AUDIO_LENGTH]

        return [
            np.ascontiguousarray(audio, dtype=np.float32)
        ]

    segments = []
    full_window_end = (len(audio) // AUDIO_LENGTH) * AUDIO_LENGTH

    # Collect complete, non-overlapping windows.
    for start in range(0, full_window_end, AUDIO_LENGTH):
        segment = audio[start:start + AUDIO_LENGTH]
        segments.append(
            np.ascontiguousarray(segment, dtype=np.float32)
        )

    # Handle any remaining samples with a full window ending at the end.
    if full_window_end < len(audio):
        final_start = len(audio) - AUDIO_LENGTH
        final_segment = audio[final_start:final_start + AUDIO_LENGTH]

        # Avoid adding an identical final window.
        if not segments or not np.array_equal(
            final_segment, segments[-1]
        ):
            segments.append(
                np.ascontiguousarray(final_segment, dtype=np.float32)
            )

    if not segments:
        raise ValueError("Could not prepare any audio segments.")

    return segments


def detect_voice(file_path: str):
    """Analyze audio and classify it using the experimental threshold."""
    model = load_aasist_model()
    segments = prepare_audio_segments(file_path)

    segment_scores = []
    inference_batch_size = 1

    for batch_start in range(0, len(segments), inference_batch_size):
        batch_segments = segments[
            batch_start:batch_start + inference_batch_size
        ]

        batch_array = np.stack(batch_segments).astype(
            np.float32, copy=False
        )

        audio_tensor = torch.from_numpy(batch_array).to(DEVICE)

        with torch.inference_mode():
            _, output = model(audio_tensor)

        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or output.shape[0] != len(batch_segments)
            or output.shape[1] < 2
        ):
            raise RuntimeError(
                "Unexpected output from the AASIST model."
            )

        batch_scores = output[:, 1].detach().cpu().numpy()

        if not np.all(np.isfinite(batch_scores)):
            raise RuntimeError(
                "AASIST returned an invalid score."
            )

        segment_scores.extend(
            float(score) for score in batch_scores
        )

    if not segment_scores:
        raise RuntimeError(
            "AASIST did not return any segment scores."
        )

    mean_score = float(np.mean(segment_scores))
    median_score = float(np.median(segment_scores))

    threshold_data = load_classification_threshold()

    classification = "NOT_CALIBRATED"
    threshold = None
    threshold_status = "NOT_AVAILABLE"

    if threshold_data is None:
        message = (
            "Analysis completed, but a valid threshold is unavailable. "
            "No real-versus-spoof classification was made."
        )
    else:
        threshold = float(threshold_data["threshold"])
        threshold_status = threshold_data["status"]

        # Higher scores mean more bona-fide-like.
        # These scores are raw model outputs, not probabilities.
        if median_score >= threshold:
            classification = "BONAFIDE"
            message = (
                "The model classifies this recording as more "
                "bona-fide-like than the experimental threshold. "
                "This does not prove the voice is genuine."
            )
        else:
            classification = "SPOOF"
            message = (
                "The model classifies this recording as more spoof-like "
                "than the experimental threshold. "
                "This does not prove the voice is cloned."
            )

    return {
        "model": "AASIST",
        "sample_rate": SAMPLE_RATE,
        "segment_duration_seconds": round(
            AUDIO_LENGTH / SAMPLE_RATE, 4
        ),
        "segments_analyzed": len(segment_scores),
        "segment_scores": [
            round(score, 6) for score in segment_scores
        ],
        "mean_bona_fide_score": round(mean_score, 6),
        "median_bona_fide_score": round(median_score, 6),
        "classification": classification,
        "threshold": (
            round(threshold, 6)
            if threshold is not None
            else None
        ),
        "threshold_status": threshold_status,
        "message": message,
        "probability_calibrated": False,
    }