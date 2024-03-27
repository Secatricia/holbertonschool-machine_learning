#!/usr/bin/env python3
"""Convolutions and Pooling"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    pooled_images = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            if mode == 'max':
                pooled_images[:, i, j, :] = np.max(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, i, j, :] = np.mean(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(1, 2))

    return pooled_images
