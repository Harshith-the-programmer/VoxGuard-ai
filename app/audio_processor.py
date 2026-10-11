
import librosa
import numpy as np


def preprocess_audio(file_path: str):
    # Load audio and convert it to mono at 16 kHz
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    # Calculate audio duration
    duration = librosa.get_duration(
        y=audio,
        sr=sample_rate
    )

    # Extract audio features
    rms = librosa.feature.rms(y=audio)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    return {
        "sample_rate": sample_rate,
        "channels": 1,
        "duration_seconds": round(float(duration), 2),
        "audio_samples": int(len(audio)),
        "average_rms": round(float(np.mean(rms)), 6),
        "average_spectral_centroid": round(
            float(np.mean(spectral_centroid)), 2
        )
    }
