import cv2
import numpy as np

def create_gabor_bank(ksize: int = 15, sigma: float = 2.0, lambd: float = 6.0, 
                      gamma: float = 0.5, psi: float = 0, num_orientations: int = 12) -> list:
    """
    Generates a bank of 2D Gabor filters across uniform orientations (theta).
    """
    filters = []
    for i in range(num_orientations):
        theta = i * np.pi / num_orientations
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
        filters.append(kern)
    return filters

def apply_gabor_mip(image: np.ndarray, filters: list) -> np.ndarray:
    """
    Filters the image through all kernels and takes the Maximum Intensity Projection (MIP)
    to capture vessels regardless of orientation.
    """
    accumulated = np.zeros_like(image, dtype=np.float32)
    for kern in filters:
        fimg = cv2.filter2D(image, cv2.CV_32F, kern)
        accumulated = np.maximum(accumulated, fimg)
    
    return np.clip(accumulated, 0, 255).astype(np.uint8)