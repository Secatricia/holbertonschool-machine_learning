#!/usr/bin/env python3
"""Convolutional Neural Networks"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """performs back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    
    dA_prev = np.zeros_like(A_prev)
    
    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for c in range(c_new):
                    vert_start = h * sh
                    vert_end = vert_start + kh
                    horiz_start = w * sw
                    horiz_end = horiz_start + kw

                    if mode == "max":
                        a_prev_slice = A_prev[
                            i, vert_start:vert_end,
                            horiz_start:horiz_end, c]
                        mask = (a_prev_slice == np.max(
                            a_prev_slice))
                        dA_prev[
                            i, vert_start:vert_end,
                            horiz_start:horiz_end,
                             c] += mask * dA[i, h, w, c]
                    elif mode == "avg":
                        da = dA[i, h, w, c]
                        shape = (kh, kw)
                        average = da / (kh * kw)
                        dA_prev[
                            i, vert_start:vert_end,
                            horiz_start:horiz_end,
                            c] += np.ones(shape) * average
    
    return dA_prev
