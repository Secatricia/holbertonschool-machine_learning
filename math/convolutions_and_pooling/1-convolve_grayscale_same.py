#!/usr/bin/env python3
"""Convolutions and Pooling"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = kh // 2
    pw = kw // 2

    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    convolved_images = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel.reshape(1, kh, kw),
                axis=(1, 2)
            )

    return convolved_images
