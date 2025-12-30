import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image

# Add parent directory to path to find utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils_config
import utils_ai

# Initialize Globals
print("Loading Models...")
utils_ai.load_models_global()

# Test on one of the failed Aang folders
FAILED_FOLDER = "debug/20251229_233408"
crop_path = f"c:/Users/kyler/Workspace/scryglass/{FAILED_FOLDER}/crop_raw.jpg"

if not os.path.exists(crop_path):
    print(f"Crop not found at {crop_path}")
    exit()

print(f"Testing Aang Crop: {crop_path}")
img = Image.open(crop_path).convert("RGB")
w, h = img.size

# Extract Art Zone (Standard)
art_x1, art_y1 = int(w * 0.08), int(h * 0.11)
art_x2, art_y2 = int(w * 0.92), int(h * 0.55)
art_img = img.crop((art_x1, art_y1, art_x2, art_y2))

# Extract Full Card
full_img = img

# Get Embeddings using utils_ai
with torch.no_grad():
    model = utils_ai.models["dino"]
    preprocess = utils_ai.preprocessors["dino"]
    device = utils_ai.device
    
    # Process Art and Full
    art_t = preprocess(art_img).unsqueeze(0).to(device)
    full_t = preprocess(full_img).unsqueeze(0).to(device)
    
    # Get Patch Maps
    art_patches = F.normalize(model.get_intermediate_layers(art_t, n=1, reshape=True)[0], p=2, dim=1)
    full_patches = F.normalize(model.get_intermediate_layers(full_t, n=1, reshape=True)[0], p=2, dim=1)

# Check against Aang in Index
target_id = "de89fec4-f5c8-4513-8504-ac9bafb44054" # Aang
if target_id not in utils_ai.art_patches:
    print("Aang not found in patch index")
    exit()
    
data = utils_ai.art_patches[target_id]
ref_art = data["art"].to(device)
ref_full = data["full"].to(device)

# Use Utils to Verify
score = utils_ai.spatial_verification(art_patches, full_patches, ref_art, ref_full)
print(f"Spatial Score for Aang: {score:.4f}")

if score > 0.32:
    print("SUCCESS: Aang Detected!")
else:
    print("FAILURE: Aang score too low.")
