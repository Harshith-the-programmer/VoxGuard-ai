import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.audio_processor import preprocess_audio
from app.aasist_detector import detect_voice


app = FastAPI(
    title="AI Voice Clone Detection API",
    description="AI-powered voice clone detection service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vox-guard-frontend.vercel.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum upload size: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024

# Supported extension -> container/format family
SUPPORTED_FORMATS = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".ogg": "ogg",
    ".oga": "ogg",
    ".opus": "ogg",
    ".m4a": "mp4",
    ".mp4": "mp4",
    ".aac": "aac",
    ".webm": "webm",
    ".aiff": "aiff",
    ".aif": "aiff",
    ".aifc": "aiff",
}


# ---------------------------------------------------------
# AUDIO VALIDATION
# ---------------------------------------------------------

def validate_audio(contents: bytes, filename: str | None) -> str:
    """Validate upload size, extension, and file signature."""

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Audio file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds the 20 MB limit.",
        )

    suffix = Path(filename or "").suffix.lower()
    audio_format = SUPPORTED_FORMATS.get(suffix)

    if audio_format is None:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported audio format. Supported formats are WAV, "
                "MP3, FLAC, OGG, OGA, OPUS, M4A, MP4, AAC, WEBM, "
                "AIFF, AIF and AIFC."
            ),
        )

    # WAV and RF64
    if audio_format == "wav":
        valid = (
            len(contents) >= 12
            and contents[:4] in (b"RIFF", b"RF64")
            and contents[8:12] == b"WAVE"
        )

    # MP3
    elif audio_format == "mp3":
        valid = (
            (len(contents) >= 10 and contents.startswith(b"ID3"))
            or (
                len(contents) >= 2
                and contents[0] == 0xFF
                and (contents[1] & 0xE0) == 0xE0
            )
        )

    # FLAC
    elif audio_format == "flac":
        valid = contents.startswith(b"fLaC")

    # OGG, OGA and OPUS commonly use the Ogg container.
    # FFmpeg performs the actual codec/decode validation.
    elif audio_format == "ogg":
        valid = contents.startswith(b"OggS")

    # WebM / Matroska EBML signature
    elif audio_format == "webm":
        valid = contents.startswith(b"\x1A\x45\xDF\xA3")

    # MP4 / M4A container
    elif audio_format == "mp4":
        valid = (
            len(contents) >= 12
            and contents[4:8] == b"ftyp"
        )

    # Raw AAC with ADTS framing
    elif audio_format == "aac":
        valid = (
            len(contents) >= 7
            and contents[0] == 0xFF
            and (contents[1] & 0xF6) == 0xF0
        )

    # AIFF and compressed AIFF-C
    elif audio_format == "aiff":
        valid = (
            len(contents) >= 12
            and contents[:4] == b"FORM"
            and contents[8:12] in (b"AIFF", b"AIFC")
        )

    else:
        valid = False

    if not valid:
        raise HTTPException(
            status_code=400,
            detail=(
                "The file contents do not match the selected audio "
                "extension. Please upload a valid audio file."
            ),
        )

    return suffix


async def read_validated_audio(file: UploadFile) -> tuple[bytes, str]:
    """Read at most 20 MB plus one byte, then validate the upload."""

    contents = await file.read(MAX_FILE_SIZE + 1)
    suffix = validate_audio(contents, file.filename)

    return contents, suffix


# ---------------------------------------------------------
# TEMPORARY FILE HANDLING
# ---------------------------------------------------------

def write_temp_audio(
    contents: bytes,
    suffix: str,
    directory: str,
) -> Path:
    """Write uploaded bytes to a temporary file."""

    temp_path = Path(directory) / f"upload{suffix}"
    temp_path.write_bytes(contents)

    if (
        not temp_path.is_file()
        or temp_path.stat().st_size != len(contents)
    ):
        raise RuntimeError("Temporary audio file creation failed.")

    return temp_path


# ---------------------------------------------------------
# FFMPEG CONVERSION
# ---------------------------------------------------------

def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """
    Decode supported input audio into mono, 16 kHz, PCM WAV.

    This allows the preprocessing and detection functions to receive
    a consistent WAV file regardless of the uploaded audio format.
    """

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "FFmpeg was not found. Install FFmpeg, add it to your "
                "system PATH, and restart the AI service."
            ),
        )

    command = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(input_path),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )

    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=422,
            detail="Audio conversion timed out. Try a shorter recording.",
        ) from None

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="Unable to start FFmpeg for audio conversion.",
        ) from None

    if completed.returncode != 0 or not output_path.is_file():
        raise HTTPException(
            status_code=422,
            detail=(
                "FFmpeg could not decode this recording. "
                "Check that the file contains valid audio and uses "
                "a supported codec."
            ),
        )

    if output_path.stat().st_size <= 44:
        raise HTTPException(
            status_code=422,
            detail="The converted audio is empty or invalid.",
        )

    return output_path


# ---------------------------------------------------------
# HEALTH ENDPOINTS
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "AI Voice Clone Detection API is running",
        "status": "UP",
    }


@app.get("/health")
def health_check():
    return {
        "message": "Python AI service is healthy",
        "status": "UP",
    }


# ---------------------------------------------------------
# AUDIO PREPROCESSING ENDPOINT
# ---------------------------------------------------------

@app.post("/api/audio/preprocess")
async def preprocess(file: UploadFile = File(...)):
    contents, suffix = await read_validated_audio(file)

    try:
        with tempfile.TemporaryDirectory(
            prefix="voxguard_audio_"
        ) as temp_dir:

            original_path = write_temp_audio(
                contents,
                suffix,
                temp_dir,
            )

            wav_path = Path(temp_dir) / "normalized.wav"

            # Convert every supported format to standard WAV.
            convert_to_wav(original_path, wav_path)

            # Run the existing preprocessing function on WAV.
            result = preprocess_audio(str(wav_path))

            return {
                "message": "Audio preprocessing successful",
                "filename": file.filename,
                "status": "PREPROCESSED",
                "features": result,
            }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=422,
            detail=(
                "Audio preprocessing failed. Please upload a valid, "
                "decodable audio recording."
            ),
        ) from None

    finally:
        await file.close()


# ---------------------------------------------------------
# AI VOICE CLONE DETECTION ENDPOINT
# ---------------------------------------------------------

@app.post("/api/audio/detect")
async def detect(file: UploadFile = File(...)):
    contents, suffix = await read_validated_audio(file)

    try:
        with tempfile.TemporaryDirectory(
            prefix="voxguard_audio_"
        ) as temp_dir:

            original_path = write_temp_audio(
                contents,
                suffix,
                temp_dir,
            )

            wav_path = Path(temp_dir) / "normalized.wav"

            # Convert OPUS, AIFF, and other supported formats to WAV.
            convert_to_wav(original_path, wav_path)

            # Run the existing AASIST detector on the converted WAV.
            result = detect_voice(str(wav_path))

            return {
                "filename": file.filename,
                "status": "ANALYSIS_COMPLETED",
                "result": result,
            }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=422,
            detail=(
                "Voice detection failed. Please upload a valid, "
                "decodable audio recording."
            ),
        ) from None

    finally:
        await file.close()