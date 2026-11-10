import cv2
import numpy as np

def calculate_vessel_density(vessel_mask: np.ndarray, fov_mask: np.ndarray) -> float:
    """
    Calculates percentage of retinal tissue area occupied by segmented blood vessels.
    """
    fov_pixels = cv2.countNonZero(fov_mask)
    if fov_pixels == 0:
        return 0.0
    vessel_pixels = cv2.countNonZero(vessel_mask)
    return (vessel_pixels / fov_pixels) * 100.0

def evaluate_segmentation(pred_mask: np.ndarray, gt_mask: np.ndarray, fov_mask: np.ndarray) -> dict:
    """
    Evaluates prediction against an expert ground-truth mask within the FOV.
    """
    valid = fov_mask > 0
    p = (pred_mask[valid] > 0).astype(np.uint8)
    g = (gt_mask[valid] > 0).astype(np.uint8)
    
    tp = np.sum((p == 1) & (g == 1))
    fp = np.sum((p == 1) & (g == 0))
    fn = np.sum((p == 0) & (g == 1))
    tn = np.sum((p == 0) & (g == 0))
    
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    iou = float(tp) / (tp + fp + fn + 1e-7)
    sensitivity = float(tp) / (tp + fn + 1e-7)
    specificity = float(tn) / (tn + fp + 1e-7)
    
    return {
        "Dice / F1": round(float(dice), 4),
        "IoU (Jaccard)": round(float(iou), 4),
        "Sensitivity": round(float(sensitivity), 4),
        "Specificity": round(float(specificity), 4)
    }