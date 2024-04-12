#!/usr/bin/env python3
"""Convolutional Neural Networks"""

import numpy as np

def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """performs forward propagation over a convolutional layer of a neural network"""
    (m, h_prev, w_prev, c_prev) = A_prev.shape

    (kh, kw, _, c_new) = W.shape

    (sh, sw) = stride

    if padding == 'same':
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        pad_h, pad_w = 0, 0

    h_out = int((h_prev + 2 * pad_h - kh) / sh) + 1
    w_out = int((w_prev + 2 * pad_w - kw) / sw) + 1

    A = np.zeros((m, h_out, w_out, c_new))

    A_prev_pad = np.pad(
        A_prev, ((0, 0),
        (pad_h, pad_h),
        (pad_w, pad_w),
        (0, 0)), mode='constant')

    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw
                A[:, i, j, k] = np.sum(A_prev_pad[
                    :, vert_start:vert_end, horiz_start:horiz_end,
                    :] * W[:, :, :, k], axis=(1, 2, 3))

    A = A + b

    A = activation(A)

    return A
