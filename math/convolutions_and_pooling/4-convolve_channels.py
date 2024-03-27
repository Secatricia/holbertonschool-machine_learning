#!/usr/bin/env python3
"""Convolutions and Pooling"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """ performs a convolution on images with channels:"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
        oh = h
        ow = w
    else:
        ph, pw = padding
        oh = (h + 2 * ph - kh) // sh + 1
        ow = (w + 2 * pw - kw) // sw + 1

    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    convolved_images = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            convolved_images[:, i, j] = np.sum(
                padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :] * kernel,
                axis=(1, 2, 3)
            )

    return convolved_images
