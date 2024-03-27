#!/usr/bin/env python3
"""Convolutions and Pooling"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    convolved_images = np.zeros((m, h - kh + 1, w - kw + 1))

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            convolved_images[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return convolved_images
