# Scryglass Session Summary: The Learned Card Dilemma
**Date:** 2025-12-31
**Topic:** Refining "User-Guided Learning" & Eliminating False Positives

## 🟢 The Wins (What Works)
1.  **UI & UX Polish**:
    *   Removed the "Model Selection" dropdown (Hardcoded to DINOv2).
    *   Relocated "Learned Cards" button to a non-intrusive spot.
    *   Added explicit confidence tags: `(Learned Card)`.
    *   Fixed encoding issues (Emojis now render correctly).
    *   Restored missing "Discord Window" button and logs.

2.  **Pipeline Stability**:
    *   The "Dynamic Gap" logic is fully wired up and functional.
    *   The backend correctly computes `AvgLearned` vs `AvgStandard`.
    *   The penalization is applied efficiently in real-time.

3.  **Visualization**:
    *   Logs now clearly show the math: `Raw=0.80 - Pen=0.20 = Final=0.60 | Main=0.35`.

## 🔴 The Unsolved Problem: "Webcam Bias"
The core issue remains: **A card learned on a webcam will always match the webcam input 10x better than the pristine digital art.**

### The Symptoms
1.  **Inflation**: A learned card gets a Global Score of ~0.85 (Perfect Match). Standard cards get ~0.40.
2.  **False Positives**: If you learn "Tidus", and show "Yuna", Tidus (Learned) scores ~0.60. Yuna (Standard) scores ~0.45. **Tidus Wins (Incorrectly).**
3.  **The Penalty Trap**:
    *   If we penalize Tidus by `-0.20`, he drops to 0.40. Yuna wins. **(Good)**.
    *   BUT, if we show Tidus (True Positive), he effectively *matches himself*. If we apply the same `-0.20` penalty, he drops below his own Standard Score, or gets lost in the noise.

### The Attempts
1.  **Static Penalty (0.85x)**: Too brittle. Lighting changes break it.
2.  **Dynamic Gap (Avg - Avg)**:
    *   **Signal Dilution**: If we compare *All* candidates, the "noise" (hundreds of 0.20 scores) destroys the average, making the Gap near zero.
    *   **Top-K Fix**: We tried comparing only the Top 5. This helped, but...
3.  **Main Score Inflation**:
    *   We tried giving the card "credit" if it matched the Main DB.
    *   **Fatal Flaw**: The "Main Score" calculation currently uses the `Global Score` from the matching result. Since the Learned Vector matched, the Global Score is huge (0.85). This pollutes the Main Score, making it artificially high (~0.65), bypassing the penalty logic.

## 🧠 Recommendations for the Next Session

### 1. Decouple the Scoring (Critical)
The "Main Score" calculation must be independent.
*   **Current**: `MainScore = (0.4 * Learned_Global_Score) + (0.6 * Spatial_Check)`
*   **Required**: `MainScore = (0.4 * ORIGINAL_VECTOR_SCORE) + (0.6 * Spatial_Check)`
*   **Action**: When verifying a learned card, you must perform a **second DINOv2 inference** (or look up the pre-computed distance) against the *Original Reference Vector*, not the Learned Vector. This gives you the "True" Standard Score to check against.

### 2. Geometric Consistency Check (RANSAC)
Scores are noisy. Geometry is truth.
*   Instead of trusting score thresholds, run a lightweight **Homography Check** (using ORB or SIFT) between the Input Crop and the Learned Image.
*   If the geometric transformation is valid (only rotation/scale, no warping), trust it.
*   If the geometry scrambles the pixels, **discard it**, no matter how high the score is.

### 3. "Deltas" not "Absolutes"
Instead of penalizing the absolute score, look at the **Score Delta**.
*   A "False Match" (Yuna vs Tidus) usually has a distinct pattern in the DINOv2 feature map compared to a "True Match" (Tidus vs Tidus), even if the scalar score is similar.
*   *Advanced*: Train a tiny classifier (Logistic Regression) on `[GlobalScore, SpatialScore, ColorDistance]` to output `IsMatch`.

### 4. Simpler Fallback: "The Jealousy Check 2.0"
If `Learned_Score` is High but `Standard_Score` (against the correct card) is Low -> **Suspicious**.
*   Why does the card look like the Learned Tidus but *not* like the Real Tidus?
*   Likely because it's NOT Tidus, it just shares the "Webcam Grain/Lighting" of the Learned Image.

**Good luck, Operator.** 🫡
