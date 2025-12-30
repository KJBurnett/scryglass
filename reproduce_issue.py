import cv2
import numpy as np
import json
import os
from scipy.spatial import distance as dist

# CONFIG
DEBUG_ID = "20251229_183026"
BASE_DIR = f"debug/{DEBUG_ID}"
FULL_IMG = f"{BASE_DIR}/full.jpg"
INFO_JSON = f"{BASE_DIR}/info.json"

if not os.path.exists(FULL_IMG):
    print(f"Error: {FULL_IMG} not found")
    exit(1)

print(f"Loading {FULL_IMG}...")
cv_image = cv2.imread(FULL_IMG)
with open(INFO_JSON, "r") as f:
    info = json.load(f)

click_x = info["click"]["x"]
click_y = info["click"]["y"]
print(f"Click: {click_x}, {click_y}")

# --- REPLICATE LOGIC ---

def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# 1. Edge Detection
gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 10, 200)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edged, kernel, iterations=2)
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

valid_contours = []
debug_img = cv_image.copy()

print(f"Total contours: {len(contours)}")

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    if area < 2000: continue
    
    if cv2.pointPolygonTest(c, (click_x, click_y), False) >= 0:
        valid_contours.append(c)
        rect = cv2.minAreaRect(c)
        print(f"Valid Contour {i}: Area={area:.1f}, Rect={rect[1]}")
        cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 2)

cv2.circle(debug_img, (click_x, click_y), 10, (0,0,255), -1)
cv2.imwrite("test_repro.jpg", debug_img)

# Scoring
candidates = []
for c in valid_contours:
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    if w == 0 or h == 0: continue
    ar = min(w, h) / max(w, h)
    area = cv2.contourArea(c)
    is_card_shaped = 0.55 < ar < 0.90
    print(f"Candidate: Area={area}, AR={ar:.2f}, CardShaped={is_card_shaped}")
    if is_card_shaped:
        candidates.append((area, c))

candidate = None
if candidates:
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidate = candidates[0][1]
    print(f"SELECTED: Area={candidates[0][0]}")
elif valid_contours:
    print("FALLBACK: Largest Area")
    candidate = max(valid_contours, key=cv2.contourArea)

if candidate is not None:
    peri = cv2.arcLength(candidate, True)
    approx = cv2.approxPolyDP(candidate, 0.02 * peri, True)
    
    print(f"Approx Points: {len(approx)}")
    
    warped = None
    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype("float32")
        # Check point ordering
        ordered = order_points(pts)
        print(f"Ordered Pts:\n{ordered}")
        warped = four_point_transform(cv_image, pts)
        print(f"Warp Result: {warped.shape}")
    
    if warped is None or warped.shape[0] < 200 or warped.shape[1] < 200:
        print("Falling back to Box Warp")
        rect = cv2.minAreaRect(candidate)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        warped = four_point_transform(cv_image, box.astype("float32"))
        print(f"Fallback Result: {warped.shape}")
        
    # --- ENFORCER ---
    h_curr, w_curr = warped.shape[:2]
    if w_curr > h_curr:
        target_size = (880, 630)
    else:
        target_size = (630, 880)
    
    print(f"Enforcing Size: {target_size} (from {w_curr}x{h_curr})")
    warped = cv2.resize(warped, target_size, interpolation=cv2.INTER_CUBIC)
    
    # --- ENHANCE ---
    warped = cv2.detailEnhance(warped, sigma_s=10, sigma_r=0.15)
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    warped = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
    cv2.imwrite("test_crop.jpg", warped)
else:
    print("No candidate found.")
