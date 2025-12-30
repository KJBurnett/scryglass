import torch
import torch.nn.functional as F
import faiss
import json
import os
from PIL import Image
from tqdm import tqdm
import clip

# Configuration
CONFIG_PATH = "config.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dinov2_vitb14" # Hardcoded for this script

print(f"Pre-computing Art Zone patches on {DEVICE}...")

# Load Model
model = torch.hub.load('facebookresearch/dinov2', MODEL_NAME).to(DEVICE)
model.eval()

# Common preprocessing (DINOv2 standard)
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Index Map
with open("index_map_dino.json", "r") as f:
    index_map = json.load(f)

patch_data = {}

for card in tqdm(index_map):
    card_id = card.get("scryfall_id")
    img_path = f"images/{card_id}.jpg"
    
    if os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            
            # Sub-Crop A: Standard Art Zone (Top Half)
            art_x1, art_y1 = int(w * 0.08), int(h * 0.11)
            art_x2, art_y2 = int(w * 0.92), int(h * 0.53)
            art_img = img.crop((art_x1, art_y1, art_x2, art_y2))
            
            # Sub-Crop B: Full Card (Low Res)
            # This captures borderless art and unique frames
            full_img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            art_tensor = preprocess(art_img).unsqueeze(0).to(DEVICE)
            full_tensor = preprocess(full_img).unsqueeze(0).to(DEVICE) # Already resized but preprocess does it again/adds norm
            
            with torch.no_grad():
                # Extract patches for both
                art_patches = model.get_intermediate_layers(art_tensor, n=1, reshape=True)
                art_norm = F.normalize(art_patches[0], p=2, dim=1).cpu()
                
                full_patches = model.get_intermediate_layers(full_tensor, n=1, reshape=True)
                full_norm = F.normalize(full_patches[0], p=2, dim=1).cpu()
                
            patch_data[card_id] = {
                "art": art_norm,
                "full": full_norm
            }
        except Exception as e:
            print(f"Error processing {card_id}: {e}")

# Save as .pt file for fast loading
torch.save(patch_data, "scryglass_art_patches.pt")
print(f"Success! Pre-computed patches for {len(patch_data)} cards saved to scryglass_art_patches.pt")
