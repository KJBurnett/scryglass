import cv2
import numpy as np
import os

def analyze_color(image_path):
    print(f"\n--- Analyzing {os.path.basename(image_path)} ---")
    img = cv2.imread(image_path)
    if img is None:
        print("Error reading image")
        return

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Calculate Histogram of Hue
    # Hue range is 0-179 in OpenCV
    hist_hue = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    
    # Normalize
    cv2.normalize(hist_hue, hist_hue, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Detect Dominant Color Ranges
    # Red: 0-10, 170-180
    # Green: 35-85
    # Blue: 85-130
    # White/Grey/Black: Low Saturation (handle separately)
    
    # Check Saturation for Achromatic (White/Grey)
    # If pixel saturation < 30, it is grey/white
    mask_achromatic = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 50, 255]))
    achromatic_ratio = cv2.countNonZero(mask_achromatic) / (img.shape[0] * img.shape[1])
    
    print(f"Achromatic (White/Grey) Ratio: {achromatic_ratio:.2f}")
    
    if achromatic_ratio > 0.40:
        print(">> Verdict: WHITE/GREY/ARTIFACT Card")
        return "WHITE"
    
    # If not achromatic, check Hue
    # Mask Blue
    mask_blue = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255]))
    blue_ratio = cv2.countNonZero(mask_blue) / (img.shape[0] * img.shape[1])
    
    print(f"Blue Ratio: {blue_ratio:.2f}")
    
    # Mask Green
    mask_green = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
    green_ratio = cv2.countNonZero(mask_green) / (img.shape[0] * img.shape[1])
    
    print(f"Green Ratio: {green_ratio:.2f}")
    
    if blue_ratio > 0.15: # Threshold for "Blue Card"
        print(">> Verdict: BLUE Card")
        return "BLUE"
    
    if green_ratio > 0.15:
        print(">> Verdict: GREEN Card")
        return "GREEN"
        
    print(">> Verdict: OTHER")
    return "OTHER"

if __name__ == "__main__":
    # Test on Rarity (Target)
    print("TESTING TARGET (RARITY)")
    analyze_color("experiments/target.png")
    
    # Compare with an Island from the database (if exists locally)
    # We might need to fetch one if not in ./images, assuming ./images populated
    island_path = "images/island.jpg" # Hypothetical
    
    # Let's scan ./images for an Island to test against
    found_island = False
    for f in os.listdir("images"):
        if "island" in f.lower() and f.endswith(".jpg"):
            analyze_color(os.path.join("images", f))
            found_island = True
            break
            
    if not found_island:
        print("No Island found in local images to compare.")
