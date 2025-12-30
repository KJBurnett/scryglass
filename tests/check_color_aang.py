import numpy as np
from PIL import Image
import colorsys
import os

# Copy logic from app.py
def get_dominant_color_type(pil_img):
    # Median-based HLS analysis
    img_small = pil_img.resize((15, 15)) 
    arr = np.array(img_small)
    med = np.median(arr, axis=(0, 1)) 
    
    r, g, b = med / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    hue = h * 360 
    
    print(f"Stats: Hue={hue:.1f}, Sat={s:.2f}, Lum={l:.2f}")
    
    # --- LIGHTING RESILIENT BUCKETS ---
    # White: Must be desaturated AND light.
    if s < 0.15 and l > 0.50: return "W" 
    # Black: Must be very dark.
    if l < 0.20: return "B" 
    
    # Hue Mapping (Adjusted for warm indoor light)
    if 40 <= hue < 75: return "W" # Map Yellow/Warm-light to White
    if 75 <= hue < 165: return "G" # Green
    if 165 <= hue < 270: return "U" # Blue / Purple
    if 270 <= hue < 330: return "U" # Violet
    if 330 <= hue or hue < 40: return "R" # Red / Orange
    
    return "C" # Unknown

# Load Aang Crop
FAILED_FOLDER = "debug/20251229_232218"
crop_path = f"c:/Users/kyler/Workspace/scryglass/{FAILED_FOLDER}/crop_raw.jpg"

if not os.path.exists(crop_path):
    print("Crop not found")
    exit()

img = Image.open(crop_path).convert("RGB")
w, h = img.size

# Extract Art Zone (Standard)
art_x1, art_y1 = int(w * 0.08), int(h * 0.11)
art_x2, art_y2 = int(w * 0.92), int(h * 0.53)
art_img = img.crop((art_x1, art_y1, art_x2, art_y2))

print(f"--- Aang Color Audit ({FAILED_FOLDER}) ---")
dom = get_dominant_color_type(art_img)
print(f"Detected Color: {dom}")

card_colors = ["U", "W"]
print(f"Card Colors: {card_colors}")

if dom in card_colors:
    print("Result: MATCH (+ Bonus)")
elif dom == "C":
    print("Result: NEUTRAL (No Bonus/Penalty)")
else:
    print("Result: MISMATCH (Penalty)")
