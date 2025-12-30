# Scryglass: Advanced MTG Card Scanner and Identifier

Scryglass is a high-performance computer vision application designed to identify Magic: The Gathering cards from a webcam or image stream. It uses a hybrid architecture combining classic Computer Vision (OpenCV) for precise card extraction and modern Deep Learning (DINOv2) for robust identification.

## Key Features

-   **Deep Learning Identification**: Uses Meta's `DINOv2` Vision Transformer to identify cards, robust to blur, angle, and lighting conditions.
-   **Dual-Zone Verification**: Checks fingerprints of both the "Art Box" and the "Full Card" to handle borderless, extended art, and standard frames simultaneously.
-   **Robust Mode**: Automatically checks 0°, 90°, 180°, and 270° rotations to handle cards upside down or sideways.
-   **Geometric "Tournament"**: Runs 3 parallel edge detection streams (Otsu, Adaptive, Canny) and picks the best card contour based on geometric perfection (Solidity & Aspect Ratio).
-   **Glare Resilience**: Uses multi-view redundancy and spatial verification to ignore localized glare.

## Project Structure

-   **`app.py`**: The main FastAPI application server. Orchestrates the pipeline.
-   **`utils_cv.py`**: Computer Vision logic (Edge detection, Contours, Perspective Warping).
-   **`utils_ai.py`**: AI Model management (DINOv2 loading, Inference, Spatial Verification).
-   **`utils_config.py`**: Configuration management.
-   **`index.html`**: The frontend debug interface.
-   **`build_patch_index.py`**: Script to generate the DINOv2 patch index from card images.

## Setup & Running

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Server**:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```
    The server will start on port 8000. It may take 10-20 seconds to load the AI models (CLIP/DINO).

3.  **Open the Frontend**:
    Open `http://localhost:8000/` in your browser.
    (Or `http://127.0.0.1:5500/index.html` if using VS Code Live Server).

## Configuration

Edit `config.json` to tune performance:
```json
{
  "device": "auto",            // "cuda" or "cpu"
  "confidence_threshold": 0.32, // Minimum score to claim a match
  "spatial_candidates": 200     // Global search depth (higher = safer but slower)
}
```

## Troubleshooting

-   **Card not finding?**: Ensure lighting is decent. Glare directly on the art box can hurt. Try "Robust Mode" by rotating the card 180 degrees.
-   **Crashes on Startup?**: Ensure you have the `scryglass_dino.index` and `scryglass_art_patches.pt` files in the root directory. If missing, run `build_patch_index.py`.
