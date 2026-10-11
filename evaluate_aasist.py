import io
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from datasets import Audio, load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
AASIST_DIR = BASE_DIR / "aasist_source"

sys.path.insert(0, str(AASIST_DIR))

from models.AASIST import Model


DEVICE = torch.device("cpu")

AUDIO_LENGTH = 64600
SAMPLE_RATE = 16000

DATASET_NAME = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"

# 400 samples from each class:
# 200 for calibration and 200 for holdout evaluation.
SAMPLES_PER_CLASS = 400
CALIBRATION_SIZE = 200

OUTPUT_CSV = BASE_DIR / "aasist_evaluation_results.csv"
THRESHOLD_JSON = BASE_DIR / "aasist_threshold.json"


# ============================================================
# 2. LOAD PRETRAINED AASIST MODEL
# ============================================================

def load_model():
    config_path = AASIST_DIR / "config" / "AASIST.conf"

    if not config_path.is_file():
        raise FileNotFoundError(
            f"AASIST configuration not found: {config_path}"
        )

    checkpoint_path = (
        AASIST_DIR / "models" / "weights" / "AASIST.pth"
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
    print("Device:", DEVICE)

    return model


# ============================================================
# 3. PREPARE AUDIO
# ============================================================

def prepare_audio(audio, original_sample_rate=SAMPLE_RATE):
    """Split the complete recording exactly as the live detector does."""
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Empty audio sample encountered.")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numeric values.")

    if int(original_sample_rate) != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=int(original_sample_rate),
            target_sr=SAMPLE_RATE,
        )

    if audio.size == 0:
        raise ValueError("Audio is empty after resampling.")

    segments = []

    for start in range(0, len(audio), AUDIO_LENGTH):
        segment = audio[start:start + AUDIO_LENGTH]

        if len(segment) < AUDIO_LENGTH:
            repeats = int(np.ceil(AUDIO_LENGTH / len(segment)))
            segment = np.tile(segment, repeats)[:AUDIO_LENGTH]

        segment = np.ascontiguousarray(segment, dtype=np.float32)
        segments.append(torch.from_numpy(segment).unsqueeze(0))

    return segments


# ============================================================
# 4. PREDICT RECORDING-LEVEL BONA-FIDE SCORE
# ============================================================
def predict_score(model, audio, sample_rate=SAMPLE_RATE):
    """Score every segment and return the median recording-level score."""
    segment_tensors = prepare_audio(audio, sample_rate)
    segment_scores = []

    for audio_tensor in segment_tensors:
        with torch.inference_mode():
            _, output = model(audio_tensor)

        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 2
            or output.shape[0] != 1
            or output.shape[1] < 2
        ):
            raise RuntimeError(
                "Unexpected output shape from the AASIST model."
            )

        score = float(output[0, 1].item())

        if not np.isfinite(score):
            raise RuntimeError("AASIST returned a non-finite score.")

        segment_scores.append(score)

    return float(np.median(segment_scores))


# ============================================================
# 5. LOAD ASVSPOOF DATASET
# ============================================================
def load_pilot_dataset():
    print("\nLoading ASVspoof 2019 LA dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="test",
        streaming=True,
    )

    # Return encoded audio bytes/path rather than decoded audio.
    # This prevents the dataset from invoking TorchCodec.
    dataset = dataset.cast_column(
        "audio",
        Audio(decode=False),
    )

    print("Dataset stream initialized.")
    return dataset


# ============================================================
# 6. READ ENCODED AUDIO AND COLLECT BALANCED SAMPLES
# ============================================================

def decode_dataset_audio(audio_data):
    """
    Decode an encoded dataset sample using SoundFile.

    With Audio(decode=False), Hugging Face normally returns
    a dictionary containing bytes and/or a path.
    """

    if not isinstance(audio_data, dict):
        raise ValueError("Unexpected encoded audio format.")

    encoded_bytes = audio_data.get("bytes")
    audio_path = audio_data.get("path")

    if encoded_bytes is not None:
        audio, sample_rate = sf.read(
            io.BytesIO(encoded_bytes),
            dtype="float32",
        )
    elif audio_path:
        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32",
        )
    else:
        raise ValueError("Dataset audio has no bytes or path.")

    # SoundFile can return multichannel audio.
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError("Decoded audio contains no samples.")

    return audio, int(sample_rate)


def collect_samples(dataset):
    genuine_samples = []
    spoof_samples = []

    # Deterministic buffer shuffle.
    dataset = dataset.shuffle(
        seed=42,
        buffer_size=10000,
    )

    print("\nCollecting genuine and spoof samples...")

    for sample in dataset:
        label_value = sample["label"]
        label_feature = dataset.features["label"]

        if hasattr(label_feature, "int2str"):
            label = label_feature.int2str(label_value)
        else:
            label = str(label_value)

        audio_data = sample["audio"]

        try:
            audio, sample_rate = decode_dataset_audio(audio_data)
        except (OSError, RuntimeError, ValueError, TypeError):
            # Skip individual malformed or unreadable audio items.
            continue

        if label == "bonafide":
            if len(genuine_samples) < SAMPLES_PER_CLASS:
                genuine_samples.append((audio, sample_rate))

        elif label == "spoof":
            if len(spoof_samples) < SAMPLES_PER_CLASS:
                spoof_samples.append((audio, sample_rate))

        total = len(genuine_samples) + len(spoof_samples)

        if total > 0 and total % 100 == 0:
            print(
                f"Collected {total}/800 samples | "
                f"Genuine: {len(genuine_samples)} | "
                f"Spoof: {len(spoof_samples)}"
            )

        if (
            len(genuine_samples) == SAMPLES_PER_CLASS
            and len(spoof_samples) == SAMPLES_PER_CLASS
        ):
            break

    if (
        len(genuine_samples) != SAMPLES_PER_CLASS
        or len(spoof_samples) != SAMPLES_PER_CLASS
    ):
        raise RuntimeError(
            "Could not collect enough readable samples. "
            f"Genuine: {len(genuine_samples)}/"
            f"{SAMPLES_PER_CLASS}, "
            f"Spoof: {len(spoof_samples)}/"
            f"{SAMPLES_PER_CLASS}. "
            "Check the internet connection and dataset availability."
        )

    print("\nBalanced dataset collected successfully!")
    return genuine_samples, spoof_samples


# ============================================================
# 7. CALCULATE SCORES
# ============================================================

def get_scores(model, samples):
    scores = []

    for index, (audio, sample_rate) in enumerate(samples):
        score = predict_score(
            model,
            audio,
            sample_rate,
        )

        scores.append(score)

        if (index + 1) % 50 == 0:
            print(
                f"Processed {index + 1}/{len(samples)} samples"
            )

    return np.asarray(scores, dtype=np.float64)


# ============================================================
# 8. CALIBRATE CLASSIFICATION THRESHOLD
# ============================================================

def calibrate_threshold(
    model,
    calibration_genuine,
    calibration_spoof,
):
    print("\n========== THRESHOLD CALIBRATION ==========")

    print("\nScoring calibration genuine samples...")
    genuine_scores = get_scores(
        model,
        calibration_genuine,
    )

    print("\nScoring calibration spoof samples...")
    spoof_scores = get_scores(
        model,
        calibration_spoof,
    )

    # Genuine = 1; spoof = 0.
    y_true = np.concatenate(
        [
            np.ones(len(genuine_scores), dtype=int),
            np.zeros(len(spoof_scores), dtype=int),
        ]
    )

    scores = np.concatenate(
        [genuine_scores, spoof_scores]
    )

    calibration_auc = roc_auc_score(y_true, scores)

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
    )

    fnr = 1.0 - tpr

    # Do not select an infinite threshold.
    finite_indices = np.flatnonzero(np.isfinite(thresholds))

    if len(finite_indices) == 0:
        raise RuntimeError(
            "Could not calculate a finite classification threshold."
        )

    eer_index = finite_indices[
        np.argmin(
            np.abs(fpr[finite_indices] - fnr[finite_indices])
        )
    ]

    threshold = float(thresholds[eer_index])

    calibration_eer = float(
        (fpr[eer_index] + fnr[eer_index]) / 2.0
    )

    print("\nCalibration results:")
    print(f"Calibration samples: {len(y_true)}")
    print(f"Calibration ROC-AUC: {calibration_auc:.4f}")
    print(f"Calibration EER: {calibration_eer * 100:.2f}%")
    print(f"Selected threshold: {threshold:.6f}")

    return {
        "threshold": threshold,
        "roc_auc": float(calibration_auc),
        "eer": calibration_eer,
    }


# ============================================================
# 9. INDEPENDENT HOLDOUT EVALUATION
# ============================================================

def evaluate_model(
    model,
    evaluation_genuine,
    evaluation_spoof,
    threshold,
):
    print("\n========== INDEPENDENT EVALUATION ==========")

    print("\nScoring evaluation genuine samples...")
    genuine_scores = get_scores(
        model,
        evaluation_genuine,
    )

    print("\nScoring evaluation spoof samples...")
    spoof_scores = get_scores(
        model,
        evaluation_spoof,
    )

    y_true = np.concatenate(
        [
            np.ones(len(genuine_scores), dtype=int),
            np.zeros(len(spoof_scores), dtype=int),
        ]
    )

    scores = np.concatenate(
        [genuine_scores, spoof_scores]
    )

    # Apply the threshold selected on calibration samples.
    y_pred = (scores >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    auc = roc_auc_score(y_true, scores)

    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr

    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else 0.0
    )

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Total samples: {len(y_true)}")
    print(f"Genuine samples: {len(genuine_scores)}")
    print(f"Spoof samples: {len(spoof_scores)}")
    print(f"\nROC-AUC: {auc:.4f}")
    print(f"EER: {eer * 100:.2f}%")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Applied calibration threshold: {threshold:.6f}")

    print("\n========== CONFUSION MATRIX ==========")
    print(f"True Positive  (Genuine predicted Genuine): {tp}")
    print(f"False Negative (Genuine predicted Spoof):   {fn}")
    print(f"False Positive (Spoof predicted Genuine):   {fp}")
    print(f"True Negative  (Spoof predicted Spoof):      {tn}")

    results = pd.DataFrame(
        {
            "actual_label": [
                "bonafide" if label == 1 else "spoof"
                for label in y_true
            ],
            "bona_fide_score": scores,
            "predicted_label": [
                "bonafide" if label == 1 else "spoof"
                for label in y_pred
            ],
        }
    )

    results.to_csv(OUTPUT_CSV, index=False)

    print(f"\nResults saved to: {OUTPUT_CSV}")

    print(
        "\nIMPORTANT: Evaluation samples were not used "
        "to select the classification threshold."
    )
    print(
        "This is an internal holdout evaluation from the "
        "ASVspoof 2019 LA test split, not an official benchmark result."
    )

    return {
        "roc_auc": float(auc),
        "eer": eer,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": specificity,
        "true_positive": int(tp),
        "false_negative": int(fn),
        "false_positive": int(fp),
        "true_negative": int(tn),
    }


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():
    print("========================================")
    print("       AASIST MODEL EVALUATION")
    print("========================================")

    model = load_model()
    dataset = load_pilot_dataset()

    genuine_samples, spoof_samples = collect_samples(dataset)

    # Keep calibration and evaluation samples separate.
    calibration_genuine = genuine_samples[:CALIBRATION_SIZE]
    evaluation_genuine = genuine_samples[CALIBRATION_SIZE:]

    calibration_spoof = spoof_samples[:CALIBRATION_SIZE]
    evaluation_spoof = spoof_samples[CALIBRATION_SIZE:]

    print("\n========== DATASET SPLIT ==========")
    print(f"Calibration genuine: {len(calibration_genuine)}")
    print(f"Calibration spoof: {len(calibration_spoof)}")
    print(f"Evaluation genuine: {len(evaluation_genuine)}")
    print(f"Evaluation spoof: {len(evaluation_spoof)}")

    calibration_results = calibrate_threshold(
        model,
        calibration_genuine,
        calibration_spoof,
    )

    threshold = calibration_results["threshold"]

    evaluation_results = evaluate_model(
        model,
        evaluation_genuine,
        evaluation_spoof,
        threshold,
    )

    # Save an experimental threshold and its evaluation metadata.
    # This file is not proof of production-level accuracy.
    threshold_data = {
        "model": "AASIST",
        "threshold": threshold,
        "positive_class": "bonafide",
        "higher_score_means": "more_bona_fide_like",
        "calibration_dataset": DATASET_NAME,
        "calibration_samples_per_class": CALIBRATION_SIZE,
        "calibration_roc_auc": calibration_results["roc_auc"],
        "calibration_eer": calibration_results["eer"],
        "holdout_evaluation": evaluation_results,
        "status": "EXPERIMENTAL_INTERNAL_HOLDOUT",
        "probability_calibrated": False,
    }

    with THRESHOLD_JSON.open("w", encoding="utf-8") as file:
        json.dump(threshold_data, file, indent=4)

    print(f"\nThreshold metadata saved to: {THRESHOLD_JSON}")

    print("\n========================================")
    print("       EVALUATION COMPLETED")
    print("========================================")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR DURING EVALUATION:")
        print(f"{type(error).__name__}: {error}")
        raise