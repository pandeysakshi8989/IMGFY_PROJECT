import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

# =====================================
# Feature Extraction & Forgery Methods
# =====================================

def apply_dct(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = np.float32(gray) / 255.0
    dct_result = cv2.dct(img_float)
    dct_norm = cv2.normalize(dct_result, None, 0, 255, cv2.NORM_MINMAX)
    dct_uint8 = np.uint8(dct_norm)
    return cv2.cvtColor(dct_uint8, cv2.COLOR_GRAY2BGR)

def apply_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX)
    lbp_uint8 = np.uint8(lbp_norm)
    return cv2.cvtColor(lbp_uint8, cv2.COLOR_GRAY2BGR)

def apply_pca(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_flat = gray.reshape(-1, 1).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(gray_flat, mean=None)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Placeholder visualization

def detect_and_draw_keypoints(img, method='sift'):
    detector = None
    if method == 'sift':
        detector = cv2.SIFT_create()
    elif method == 'orb':
        detector = cv2.ORB_create()
    elif method == 'brisk':
        detector = cv2.BRISK_create()
    elif method == 'surf':
        if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SURF_create'):
            detector = cv2.xfeatures2d.SURF_create()
        else:
            raise ImportError("Install opencv-contrib-python for SURF.")
    elif method == 'kaze':
        detector = cv2.KAZE_create()
    else:
        raise ValueError(f"Unsupported method: {method}")

    keypoints, _ = detector.detectAndCompute(img, None)
    return cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# =====================================
# Main API-like Function (Modular Use)
# =====================================

def detect_forgery(image_path, algorithm):
    """
    Process an image with the selected forgery detection algorithm.
    
    Args:
        image_path (str): Path to the image file.
        algorithm (str): Algorithm to apply (e.g., sift, dct, lbp, etc.)
    
    Returns:
        np.ndarray: The processed image (OpenCV format).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    algorithm = algorithm.lower()

    try:
        if algorithm in ['sift', 'orb', 'brisk', 'surf', 'kaze']:
            return detect_and_draw_keypoints(image, method=algorithm)
        elif algorithm == 'dct':
            return apply_dct(image)
        elif algorithm == 'lbp':
            return apply_lbp(image)
        elif algorithm == 'pca':
            return apply_pca(image)
        elif algorithm in ['svm', 'cnn', 'autoencoders', 'decision_tree', 'random_forest']:
            print(f"[INFO] ML model '{algorithm}' is a placeholder. Add actual model loading and inference.")
            return image  # Placeholder
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    except Exception as e:
        print(f"[ERROR] Failed to process with algorithm '{algorithm}': {e}")
        return image  # Return original image on failure
