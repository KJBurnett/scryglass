import cv2
import numpy as np
import json
import os
from scipy.spatial import distance as dist
from PIL import Image

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute width and height to decide orientation
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # ASPECT RATIO SNAPPING: Force exactly 630x880 (Magic Card Ratio)
    # This fixes "fat" or "thin" detection errors.
    is_landscape = maxWidth > maxHeight
    if is_landscape:
        targetW, targetH = 880, 630
    else:
        targetW, targetH = 630, 880

    dst = np.array([
        [0, 0],
        [targetW - 1, 0],
        [targetW - 1, targetH - 1],
        [0, targetH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (targetW, targetH))
    return warped

def expand_box(box, scale=1.15):
    # box is 4x2 array
    center = np.mean(box, axis=0)
    # vectors from center to corners
    vectors = box - center
    # scale them
    expanded_vectors = vectors * scale
    # new corners
    new_box = center + expanded_vectors
    return new_box.astype('float32') # Warp expects float32

def find_card_contour(cv_image, click_point, debug_dir, debug_info):
    # 1. Multi-Stream Edge Detection
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    streams = []
    
    # Stream A: Otsu (Best for separating card body from dark mats)
    blurred_a = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_otsu = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    otsu_edges = cv2.morphologyEx(otsu, cv2.MORPH_GRADIENT, kernel_otsu)
    streams.append(("otsu", otsu_edges))
    
    # Stream B: Adaptive Threshold (Best for uneven lighting/glare)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    adaptive = cv2.bitwise_not(adaptive)
    streams.append(("adaptive", adaptive))
    
    # Stream C: Canny (Sharp lines)
    clahe_detector = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_en = clahe_detector.apply(gray)
    blurred_c = cv2.GaussianBlur(gray_en, (5, 5), 0)
    canny = cv2.Canny(blurred_c, 50, 200)
    streams.append(("canny", canny))
    
    # TOURNAMENT: Find Best Card Shape across all streams
    all_card_candidates = []
    debug_contours = cv_image.copy()
    
    for stream_name, edge_map in streams:
        # Dilate stream independently to close borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_map = cv2.dilate(edge_map, kernel, iterations=1)
        
        cnts, _ = cv2.findContours(edge_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in cnts:
            if cv2.contourArea(c) < 2000: continue
            if cv2.pointPolygonTest(c, click_point, False) < 0: continue
            
            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0: continue
            
            ar = min(w, h) / max(w, h)
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # MAGIC CARD AR: 0.716
            # STRICTNESS: 10x penalty for AR mismatch
            target_ar = 0.716
            ar_score = max(0, 1.0 - abs(ar - target_ar) * 10.0) 
            
            # STRICTNESS: Exponential solidity (0.95 vs 0.99 is massive)
            solidity_score = solidity ** 20
            
            # TOTAL GEOMETRIC SCORE (Ignore area, focus on SHAPE)
            geom_score = ar_score * solidity_score
            
            # Filter: Must at least look VAGUELY like a card to be in the running
            if ar_score > 0.1 and solidity > 0.90:
                all_card_candidates.append({
                    "stream": stream_name,
                    "cnt": c,
                    "geom_score": geom_score,
                    "ar": ar,
                    "solidity": solidity,
                    "area": area
                })
                # Draw candidate in Yellow
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(debug_contours, [box], 0, (0, 255, 255), 2)

    cv2.imwrite(f"{debug_dir}/tournament_candidates.jpg", debug_contours)

    # SELECTION: Best Card-Shape Wins
    if all_card_candidates:
        all_card_candidates.sort(key=lambda x: x["geom_score"], reverse=True)
        best = all_card_candidates[0]
        debug_info["geometric_metrics"] = {
            "stream": best["stream"],
            "ar": best["ar"],
            "solidity": best["solidity"],
            "geom_score": best["geom_score"],
            "area": best["area"]
        }
        print(f"Selection Winner: {best['stream']} Stream (AR: {best['ar']:.3f}, Sol: {best['solidity']:.3f}, Score: {best['geom_score']:.3f})")
        return best["cnt"]
    else:
        return None

def enhance_image(warped):
    # SHARPEN
    warped = cv2.detailEnhance(warped, sigma_s=10, sigma_r=0.15)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        warped = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    except Exception as e:
        print(f"Enhancement failed: {e}")
    return warped
