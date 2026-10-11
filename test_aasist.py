import json
import sys
from pathlib import Path

import numpy as np
import torch

# Locate official AASIST source
BASE_DIR = Path(__file__).resolve().parent
AASIST_DIR = BASE_DIR / "aasist_source"

sys.path.insert(0, str(AASIST_DIR))

from models.AASIST import Model

# CPU inference
device = torch.device("cpu")

# Load model configuration
config_path = AASIST_DIR / "config" / "AASIST.conf"

with open(config_path, "r") as file:
    config = json.load(file)

model_config = config["model_config"]

# Initialize model
model = Model(model_config).to(device)

# Load pretrained checkpoint
checkpoint_path = (
    AASIST_DIR / "models" / "weights" / "AASIST.pth"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint, strict=True)
model.eval()

print("AASIST checkpoint loaded successfully!")
print("Device:", device)

# Create test audio (4.0375 seconds at 16 kHz)
test_audio = np.random.randn(64600).astype(np.float32)

# Convert to PyTorch tensor
audio_tensor = torch.from_numpy(test_audio).unsqueeze(0)

# Run inference
with torch.no_grad():
    hidden, output = model(audio_tensor)

    bona_fide_score = output[:, 1].item()

print("\nInference successful!")
print("Hidden output shape:", tuple(hidden.shape))
print("Model output shape:", tuple(output.shape))
print("Bona-fide score:", bona_fide_score)