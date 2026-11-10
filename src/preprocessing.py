import cv2
import numpy as np

def extract_green_channel(bgr_image: np.ndarray) -> np.ndarray:
    """
    Extracts the green color channel (index 1 in BGR) where hemoglobin
    exhibits maximal light absorption.
    """
    return bgr_image[:, :, 1]

def apply_clahe(gray_image: np.ndarray, clip_limit: float = 2.5, tile_size: int = 8) -> np.ndarray:
    """
    Enhances local micro-contrast without blowing out background noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray_image)

def subtract_background_illumination(enhanced_image: np.ndarray, blur_ksize: int = 29) -> np.ndarray:
    """
    Estimates slowly varying background illumination with a wide Gaussian kernel
    and subtracts it from the image to eliminate vignetting.
    """
    ksize = blur_ksize if blur_ksize % 2 != 0 else blur_ksize + 1
    bg = cv2.GaussianBlur(enhanced_image, (ksize, ksize), 0)
    return cv2.subtract(bg, enhanced_image)