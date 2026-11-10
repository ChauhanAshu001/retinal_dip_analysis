import cv2
import argparse
from src.fov import compute_fov_mask
from src.vessels import segment_retinal_vessels
from src.exudates import detect_hard_exudates
from src.metrics import calculate_vessel_density, evaluate_segmentation
from src.visualization import build_diagnostic_overlay

'''
Use this script for quick command-line verification or batch testing without 
starting the streamlit app or browser.

This main.py expects an image path
        python main.py --image "image_path/image.jpg"
        
The flow of execution is:

    
    Read image from disk
    ↓
    compute_fov_mask()
    ↓
    segment_retinal_vessels()
    ↓
    detect_hard_exudates()
    ↓
    calculate metrics
    ↓
    Save output images

Output produced is:
    output_overlay.png
    output_vessels.png
    output_exudates.png
    
and print statistics like this to the console:
    Vessel Density: 12.45%
    Candidate Count: 8
    Dice: 0.78
    IoU: 0.65
'''


def run_pipeline(image_path: str, gt_path: str = None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    print("--> 1. Computing FOV...")
    fov = compute_fov_mask(img)
    
    print("--> 2. Segmenting Blood Vessels...")
    vessels, _ = segment_retinal_vessels(img, fov)
    density = calculate_vessel_density(vessels, fov)
    print(f"    Vessel Density: {density:.2f}%")
    
    print("--> 3. Detecting Exudates...")
    exudates, candidates, _ = detect_hard_exudates(img, fov)
    print(f"    Candidate Count: {len(candidates)}")
    
    if gt_path:
        print("--> 4. Computing Ground-Truth Evaluation...")
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt.shape != vessels.shape:
            gt = cv2.resize(gt, (vessels.shape[1], vessels.shape[0]), interpolation=cv2.INTER_NEAREST)
        metrics = evaluate_segmentation(vessels, gt, fov)
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    overlay = build_diagnostic_overlay(img, vessels, exudates)
    cv2.imwrite("output_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("output_vessels.png", vessels)
    cv2.imwrite("output_exudates.png", exudates)
    print("✓ Output images saved to current directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retinal DIP Pipeline CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal image")
    parser.add_argument("--gt", type=str, default=None, help="Optional path to ground truth mask")
    args = parser.parse_args()
    
    run_pipeline(args.image, args.gt)