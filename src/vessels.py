import cv2
import numpy as np
from src.preprocessing import extract_green_channel, apply_clahe, subtract_background_illumination
from src.gabor import create_gabor_bank, apply_gabor_mip

def segment_retinal_vessels(bgr_image: np.ndarray, fov_mask: np.ndarray, 
                            clahe_clip: float = 2.5, gabor_sigma: float = 2.0, 
                            gabor_lambda: float = 6.0) -> tuple[np.ndarray, dict]:
    """
    Executes the full pipeline for blood vessel segmentation:
    Green Channel -> CLAHE -> Illumination Subtraction -> Gabor MIP -> Otsu -> Morphology
    """
    # 1. Channel selection
    green = extract_green_channel(bgr_image)
    
    # 2. Local contrast enhancement
    green_clahe = apply_clahe(green, clip_limit=clahe_clip)
    
    # 3. Vignetting correction
    corrected = subtract_background_illumination(green_clahe)
    
    # 4. Gabor 12-orientation filtering
    filters = create_gabor_bank(ksize=15, sigma=gabor_sigma, lambd=gabor_lambda, num_orientations=12)
    gabor_response = apply_gabor_mip(corrected, filters)
    
    # 5. Otsu thresholding inside FOV
    gabor_fov = cv2.bitwise_and(gabor_response, gabor_response, mask=fov_mask)
    _, vessel_mask = cv2.threshold(gabor_fov, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 6. Morphological cleanup (Opening removes noise, Closing joins small gaps)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
    final_vessels = cv2.bitwise_and(cleaned, cleaned, mask=fov_mask)
    
    stages = {
        "Green Channel": green,
        "CLAHE Enhanced": green_clahe,
        "Background Subtracted": corrected,
        "Gabor MIP Response": gabor_response,
        "Binary Vessel Mask": final_vessels
    }
    return final_vessels, stages