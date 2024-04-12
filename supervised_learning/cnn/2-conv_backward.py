#!/usr/bin/env python3
"""Convolutional Neural Networks"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """performs back propagation over a convolutional layer of a neural network"""
    (m, h_new, w_new, c_new) = dZ.shape

    (m, h_prev, w_prev, c_prev) = A_prev.shape

    (kh, kw, _, _) = W.shape

    (sh, sw) = stride

    dA_prev = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    if padding == "same":
        pad_h = int(((h_prev - 1) * sh + kh - h_prev) / 2)
        pad_w = int(((w_prev - 1) * sw + kw - w_prev) / 2)
        A_prev_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h), (
            pad_w, pad_w), (0, 0)), mode='constant')
        dA_prev_pad = np.pad(dA_prev, ((0, 0), (pad_h, pad_h), (
            pad_w, pad_w), (0, 0)), mode='constant')
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[
                        :, :, :, c] * dZ[i, h, w, c]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
                    db[:, :, :, c] += dZ[i, h, w, c]

        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db
