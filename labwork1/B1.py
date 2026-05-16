#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np
from numpy.typing import NDArray

def grayscale(image):
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray.astype(np.uint8)

def histogram(gray, h, w):
    hist = compute_histogram(gray, h, w)
    print_histogram(hist)

def compute_histogram(gray, h, w):
    hist = np.zeros(256)
    for y in range(h):
        for x in range(w):
            hist[gray[y, x]] += 1
    return hist


def print_histogram(hist: NDArray):
    max_count = hist.max()
    screen_w = 80
    scaler = (screen_w / max_count)
    for intensity in range(256):
        bar_len = int(hist[intensity] * scaler)
        bar = "#" * bar_len
        print(f"{intensity:03d} | {bar}")

def ehistogram(gray, h, w):
    equalized = equalize_histogram(gray, h, w)
    hist = compute_histogram(equalized, h, w)
    print_histogram(hist)
    return equalized


def equalize_histogram(gray, h, w):
    hist = compute_histogram(gray, h, w)
    cdf = np.zeros(256)
    equalized = np.zeros((h,w), dtype=np.uint8)
    cdf[0] = hist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + hist[i]

    total_pixels = h * w
    new_i = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        mapped = round((cdf[i]) / total_pixels * 255)
        if mapped < 0:
            mapped = 0
        if mapped > 255:
            mapped = 255
        new_i[i] = int(mapped)

    for y in range(h):
        for x in range(w):
            equalized[y,x] = new_i[gray[y, x]]
    return equalized

def resize(gray, h, w):
    new_w = int(round(w * 1.5))
    new_h = int(round(h * 1.5))
    resized = cv2.resize(gray, (new_w, new_h))
    return resized

def print_info(image, h, w):
    if image.ndim == 2:
        channels = 1
    else:
        channels = 3
    dtype = str(image.dtype)
    min_val = int(np.min(image))
    max_val = int(np.max(image))
    print(f"Size: {h} x {w}")
    print(f"Channels: {channels}")
    print(f"Data type: {dtype}")
    print(f"Min pixel value: {min_val}")
    print(f"Max pixel value: {max_val}")

def contrast_control(gray, contrast):
    adjusted = np.clip(gray * contrast, 0, 255).astype(np.uint8)
    return adjusted

def brightness_control(gray, brightness):
    adjusted = np.clip(gray + brightness, 0, 255).astype(np.uint8)
    return adjusted

def main():
    usage = \
"Usage: python B1.py <subcommand> <image_path>\n\
Subcommands:\n\
histogram: display histogram of the grayscale image \n\
ehistogram: apply histogram equalization to the grayscale image and display its histogram \n\
resize: resize to 150% of its original width and height \n\
info: print info size, number of channels, data type, min/max pixel valeu"

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    subcommand = sys.argv[1]
    img_path = sys.argv[-1]

    input_img = cv2.imread(img_path)
    gray = grayscale(input_img)
    h= gray.shape[0]
    w= gray.shape[1]

    i_name, ext = os.path.splitext(img_path)
    gray_path = f"grayscaled_{i_name}{ext}"
    cv2.imwrite(gray_path, gray)

    if subcommand == "help":
        print(usage)
    elif subcommand == "histogram":
        histogram(gray, h, w)
    elif subcommand == "ehistogram":
        equalized = ehistogram(gray, h, w)
        out_path = f"equalized_{i_name}{ext}"
        cv2.imwrite(out_path, equalized)
    elif subcommand == "resize":
        resized = resize(gray, h, w)
        out_path = f"resized_{i_name}{ext}"
        cv2.imwrite(out_path, resized)
    elif subcommand == "info":
        print_info(input_img, h, w)
    else:
        print(f"invalid subcommand {subcommand}")
        print(usage)
        sys.exit(1)

if __name__ == "__main__":
    main()
