"""Quick DINOv2 test on latest crop"""
import torch
from PIL import Image
from torchvision import transforms
import faiss
import json
import numpy as np

# Load DINOv2
print("Loading DINOv2...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load index
index = faiss.read_index('scryglass_dino.index')
with open('index_map_dino.json', 'r') as f:
    index_map = json.load(f)

# Test latest crop
img = Image.open('experiments/latest_crop.jpg').convert('RGB')
print(f'Image size: {img.size}')

tensor = preprocess(img).unsqueeze(0)
with torch.no_grad():
    features = model(tensor)
    features = features / features.norm(dim=-1, keepdim=True)

vector = features.numpy().astype('float32')
D, I = index.search(vector, 10)

print('\n=== DINOv2 on LATEST CROP ===')
for k in range(10):
    idx = int(I[0][k])
    score = float(D[0][k])
    card = index_map[idx]
    print(f"#{k+1}: {score:.4f} - {card['name']}")

# Find Rarity's rank
rarity_idx = None
for i, card in enumerate(index_map):
    if 'Rarity' in card['name']:
        rarity_idx = i
        break

if rarity_idx is not None:
    D2, I2 = index.search(vector, 100)
    for j in range(100):
        if I2[0][j] == rarity_idx:
            print(f"\nRarity is at rank #{j+1} with score {D2[0][j]:.4f}")
            break
    else:
        print("\nRarity not in top 100!")
