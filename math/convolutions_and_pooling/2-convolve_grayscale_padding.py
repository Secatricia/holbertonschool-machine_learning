#!/usr/bin/env python3
"""Convolutions and Pooling"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)),
        mode='constant')

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    convolved_images = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel.reshape(1, kh, kw),
                axis=(1, 2)
            )

    return convolved_images
