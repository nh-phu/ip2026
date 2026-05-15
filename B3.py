#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np

def grayscale(image):
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray.astype(np.uint8)

def binary_img(gray, threshold):
    bi = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return bi

def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    img_path = sys.argv[-1]

    input_img = cv2.imread(img_path)
    gray = grayscale(input_img)

    i_name, ext = os.path.splitext(img_path)
    gray_path = f"grayscaled_{i_name}{ext}"
    cv2.imwrite(gray_path, gray)

    threshold = 80
    bc = binary_img(gray, threshold)
    out_path = f"bi_{i_name}{ext}"
    cv2.imwrite(out_path, bc)

if __name__ == "__main__":
    main()
