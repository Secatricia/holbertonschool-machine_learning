#!/usr/bin/env python3
"""Convolutional Neural Networks"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs forward propagation over a pooling layer of a neural network"""

    (m, h_prev, w_prev, c_prev) = A_prev.shape

    (kh, kw) = kernel_shape
    (sh, sw) = stride

    h_out = int(1 + (h_prev - kh) / sh)
    w_out = int(1 + (w_prev - kw) / sw)

    A = np.zeros((m, h_out, w_out, c_prev))

    for i in range(m):
        for h in range(h_out):
            for w in range(w_out):
                vert_start = h * sh
                vert_end = vert_start + kh
                horiz_start = w * sw
                horiz_end = horiz_start + kw

                a_slice_prev = A_prev[
                    i, vert_start:vert_end, horiz_start:horiz_end, :]

                if mode == 'max':
                    A[i, h, w, :] = np.max(a_slice_prev, axis=(0, 1))
                elif mode == 'avg':
                    A[i, h, w, :] = np.mean(a_slice_prev, axis=(0, 1))

    return A
