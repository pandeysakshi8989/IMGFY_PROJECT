import cv2
import numpy as np
import os

def processing_image(image_path, operation):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    operation = operation.lower()
    operation_map = {
        'grayscale': to_grayscale,
        'red_channel': red_channel,
        'green_channel': green_channel,
        'blue_channel': blue_channel,
        'contrast': enhance_contrast,
        'histogram_equalization': histogram_equalization,
        'smooth_filter': smooth_filter,
        'edge_detection': edge_detection,
        'noise_removal': noise_removal,
        'image_restoration': image_restoration,
        'segmentation': segmentation,
    }

    func = operation_map.get(operation)
    if not func:
        raise ValueError(f"Unsupported operation: {operation}")

    return func(image)

# Image Processing Tools
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def red_channel(img):
    red = np.zeros_like(img)
    red[:, :, 2] = img[:, :, 2]
    return red

def green_channel(img):
    green = np.zeros_like(img)
    green[:, :, 1] = img[:, :, 1]
    return green

def blue_channel(img):
    blue = np.zeros_like(img)
    blue[:, :, 0] = img[:, :, 0]
    return blue

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def histogram_equalization(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def smooth_filter(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def edge_detection(img):
    gray = to_grayscale(img)
    return cv2.Canny(gray, 100, 200)

def noise_removal(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def image_restoration(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

def segmentation(img):
    gray = to_grayscale(img)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(img, img, mask=mask)
