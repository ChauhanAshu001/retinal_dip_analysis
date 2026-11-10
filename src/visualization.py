import cv2
import numpy as np

def build_diagnostic_overlay(original_bgr: np.ndarray, vessel_mask: np.ndarray, 
                             exudate_mask: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """
    Overlays segmented blood vessels in Green [0, 255, 0] and 
    hard exudates in Yellow [255, 235, 0] over the original color photo.
    """
    rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    annotated = rgb.copy()
    
    # Tint blood vessels Green
    annotated[vessel_mask > 0] = [0, 255, 0]
    
    # Tint exudates Yellow
    annotated[exudate_mask > 0] = [255, 235, 0]
    
    # Alpha blend with original
    blended = cv2.addWeighted(rgb, alpha, annotated, 1.0 - alpha, 0)
    return blended