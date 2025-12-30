import cv2
import json
import os
import glob
import numpy as np

# CONFIG
TEST_CROP = "test_crop.jpg"
IMAGES_DIR = "images/"
INDEX_MAP = "index_map.json"

if not os.path.exists(TEST_CROP):
    print(f"Error: {TEST_CROP} not found")
    exit(1)

print(f"Loading {TEST_CROP}...")
img_query = cv2.imread(TEST_CROP, cv2.IMREAD_GRAYSCALE)

# Init ORB
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img_query, None)
print(f"Query Keypoints: {len(kp1)}")

if des1 is None:
    print("Error: No keypoints found in query image!")
    exit(1)

# Load Index Map to get names
with open(INDEX_MAP, 'r') as f:
    index_data = json.load(f)

# Map UUID filename to Card Name
uuid_to_name = {}
for card in index_data:
    # Extract filename from path
    fname = os.path.basename(card['image_path'])
    uuid_to_name[fname] = card['name']

# Init Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

best_match_name = None
best_match_count = 0
matches_data = []

print("Scanning local database (ORB)...")

# We will scan ALL images in images/ dir
image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))

for img_path in image_files:
    fname = os.path.basename(img_path)
    card_name = uuid_to_name.get(fname, "Unknown")
    
    img_train = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_train is None: continue
    
    # Detect
    kp2, des2 = orb.detectAndCompute(img_train, None)
    if des2 is None: continue
    
    # Match
    matches = bf.match(des1, des2)
    
    # Sort
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Heuristic: Count matches with distance < 50 (Very close features)
    good_matches = [m for m in matches if m.distance < 60]
    count = len(good_matches)
    
    matches_data.append((count, card_name))
    
    if count > best_match_count:
        best_match_count = count
        best_match_name = card_name
        # Optional: Print new leader
        # print(f"New Leader: {card_name} ({count} matches)")

matches_data.sort(key=lambda x: x[0], reverse=True)

print("\n--- ORB RESULTS (Top 10) ---")
for i in range(min(10, len(matches_data))):
    count, name = matches_data[i]
    print(f"#{i+1}: {name} - {count} matches")

print(f"\nWinner: {best_match_name} with {best_match_count} matches.")
