import torch
import clip
from PIL import Image
import json
import os

# Config
IMAGE_PATH = "test_crop.jpg"
INDEX_MAP = "index_map.json"
MODEL_NAME = "ViT-L/14"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP {MODEL_NAME} on {device}...")
model, preprocess = clip.load(MODEL_NAME, device=device)

# Load Image
print(f"Loading {IMAGE_PATH}...")
if not os.path.exists(IMAGE_PATH):
    print("Image not found!")
    exit(1)
    
image = preprocess(Image.open(IMAGE_PATH)).unsqueeze(0).to(device)

# Load Card Names
with open(INDEX_MAP, 'r') as f:
    cards = json.load(f)
    
names = [c["name"] for c in cards]
# Remove duplicates while preserving order
unique_names = list(dict.fromkeys(names))

print(f"Encoding {len(unique_names)} unique labels...")

# Create text prompts
# "A Magic the Gathering card: [Name]" works better than just "[Name]"
text_prompts = [f"A Magic the Gathering card showing {name}" for name in unique_names]
# Also try simpler names
# text_prompts = unique_names

text_inputs = clip.tokenize(text_prompts).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(10)

print("\n--- ZERO-SHOT TEXT MATCHING ---")
for value, index in zip(values, indices):
    name = unique_names[index]
    print(f"{name:>30}: {100 * value.item():.2f}%")
