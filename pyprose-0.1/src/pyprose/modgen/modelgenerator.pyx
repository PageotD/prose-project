#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: cinterp2d.pyx
#   Author: Damien Pageot
#    Email: nessi.develop@protonmail.com
#
# Copyright (C) 2019 Damien Pageot
# ------------------------------------------------------------------

"""
Functions interpolate 2D scattered data.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt

ctypedef np.float32_t DTYPE_f

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef apply_reflective_gaussian_smoothing(np.ndarray[DTYPE_f, ndim=2] model, float sigma):
    cdef int n1 = model.shape[0]
    cdef int n2 = model.shape[1]
    cdef int i1, i2, j1, j2
    cdef float sum_weights, weight, total_weight
    cdef np.ndarray[DTYPE_f, ndim=2] smoothed_model = np.zeros_like(model)

    cdef int ksize = int(2 * np.ceil(2 * sigma) + 1)
    cdef int kcenter = ksize // 2

    cdef np.ndarray[DTYPE_f, ndim=2] kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cdef float norm_factor = 0.0

    # Generate the Gaussian kernel
    for i1 in range(ksize):
        for i2 in range(ksize):
            x = i2 - kcenter
            z = i1 - kcenter
            kernel[i1, i2] = np.exp(-(x ** 2 + z ** 2) / (2 * sigma ** 2))
            norm_factor += kernel[i1, i2]

    # Normalize the kernel
    kernel /= norm_factor

    # Apply the reflective Gaussian smoothing
    for i1 in range(n1):
        for i2 in range(n2):
            sum_weights = 0.0
            total_weight = 0.0
            for j1 in range(ksize):
                for j2 in range(ksize):
                    x = i2 + j2 - kcenter
                    z = i1 + j1 - kcenter
                    if x < 0:
                        x = -x
                    elif x >= n2:
                        x = 2 * n2 - 1 - x
                    if z < 0:
                        z = -z
                    elif z >= n1:
                        z = 2 * n1 - 1 - z
                    weight = kernel[j1, j2]
                    smoothed_model[i1, i2] += weight * model[z, x]
                    sum_weights += weight
            smoothed_model[i1, i2] /= sum_weights

    return smoothed_model
    
cpdef np.ndarray[DTYPE_f, ndim=2] cython_gaussian_smoothing(np.ndarray[DTYPE_f, ndim=2] model, float sigma):
    """
    Apply Gaussian smoothing to a 2D model.

    :param model: 2D model array
    :param sigma: standard deviation of the Gaussian kernel
    :return: smoothed model
    """
    cdef int size = int(2 * np.ceil(3 * sigma) + 1)  # Size of the kernel
    cdef np.ndarray[DTYPE_f, ndim=2] smoothed_model = np.zeros_like(model, dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] kernel = np.zeros((size, size), dtype=np.float32)

    # Create the Gaussian kernel
    cdef int i, j
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((i - size//2)**2 + (j - size//2)**2) / (2 * sigma**2))

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Perform the convolution
    cdef int m, n, k, l, p, q
    cdef float value
    for m in range(model.shape[0]):
        for n in range(model.shape[1]):
            value = 0.0
            for k in range(size):
                for l in range(size):
                    p = m - size//2 + k
                    q = n - size//2 + l
                    if p >= 0 and p < model.shape[0] and q >= 0 and q < model.shape[1]:
                        value += model[p, q] * kernel[k, l]
            smoothed_model[m, n] = value

    return smoothed_model

cpdef cneighbor(n1, n2, dh, xp, zp, val):
    """
    Natural Neighbor 2D interpolation.

    :param n1: number of points in the first dimension of the output model
    :param n2: number of points in the second dimension of the output model
    :param dh: spatial sampling
    :param xp: x-coordinates of points to interpolate
    :param zp: z-coordinates of points to interpolate
    :param val: value of the points to interpolate
    """

    # Get the number of points
    npts = xp.size

    # Declare output array
    model = np.zeros((n1, n2), dtype=np.float32)

    # Loop over dimensions
    for i2 in range(n2):
        x = float(i2) * dh
        for i1 in range(n1):
            z = float(i1) * dh

            # Initialize variables
            numerator_sum = 0.0
            denominator_sum = 0.0

            # Loop over number of scatter points
            for ipts in range(npts):
                # Calculate distance
                d = sqrt((x - xp[ipts])**2 + (z - zp[ipts])**2)

                # Check if distance is zero
                if d == 0.0:
                    model[i1, i2] = val[ipts]
                    break

                # Calculate weights using inverse distance
                weight = 1.0 / d

                # Accumulate numerator and denominator
                numerator_sum += weight * val[ipts]
                denominator_sum += weight

            # Interpolate value using weighted average
            if model[i1, i2] == 0.0:
                model[i1, i2] = numerator_sum / denominator_sum

    return model

cpdef cvoronoi(int n1, int n2, float dh, np.ndarray[DTYPE_f, ndim=1] xp, np.ndarray[DTYPE_f, ndim=1] zp, np.ndarray[DTYPE_f, ndim=1] val):
    """
    Simple Voronoi 2D interpolation.

    :param n1: number of points in the first dimension of the output model
    :param n2: number of points in the second dimension of the output model
    :param dh: spatial sampling
    :param xp: x-coordinates of points to interpolate
    :param zp: z-coordinates of points to interpolate
    :param val: value of the points to interpolate
    """

    cdef Py_ssize_t i1, i2, ipts

    # Get the number of points
    cdef int npts = np.size(xp)

    # Declare output array
    cdef np.ndarray[DTYPE_f, ndim=2] model = np.zeros((n1, n2), dtype=np.float32)

    # Decalre variables
    cdef float x, z, d, dmin

    # Loop over dimensions
    for i2 in range(0, n2):
        x = float(i2)*dh
        for i1 in range(0, n1):
            z = float(i1)*dh
            # Initialize minimum distance
            dmin = 0.
            # Loop over number of scatter points
            for ipts in range(0, npts):
                # Calculate distance
                d = sqrt((x-xp[ipts])**2+(z-zp[ipts])**2)
                if ipts == 0:
                    dmin = d
                    imin = ipts
                else:
                    if d < dmin:
                        dmin = d
                        imin = ipts
            # Fill output model
            model[i1, i2] = val[imin]

    return model

cpdef cinvdist(int n1, int n2, float dh, int pw, np.ndarray[DTYPE_f, ndim=1] xp, np.ndarray[DTYPE_f, ndim=1] zp, np.ndarray[DTYPE_f, ndim=1] val):
    """
    Inverse distance weightning 2D interpolation.

    :param n1: number of points in the first dimension of the output model
    :param n2: number of points in the second dimension of the output model
    :param dh: spatial sampling
    :param pw: power to apply
    :param xp: x-coordinates of points to interpolate
    :param zp: z-coordinates of points to interpolate
    :param val: value of the points to interpolate
    """

    cdef Py_ssize_t i1, i2, ipts

    # Get the number of points
    cdef int npts = np.size(xp)

    # Declare output array
    cdef np.ndarray[DTYPE_f, ndim=2] model = np.zeros((n1, n2), dtype=np.float32)

    # Decalre variables
    cdef float x, z, num, den, d, w

    # Loop over dimensions
    for i2 in range(0, n2):
        x = float(i2)*dh
        for i1 in range(0, n1):
            z = float(i1)*dh
            # Initialize numerator and denominator
            num = 0.
            den = 0.
            # Loop over number of scatter points
            for ipts in range(0, npts):
                # Calculate distances
                d = sqrt((x-xp[ipts])**2+(z-zp[ipts])**2)
                if pow(d, pw) > 0.:
                    w = 1./pow(d, pw)
                    num = num+w*val[ipts]
                    den = den+w
                else:
                    num = val[ipts]
                    den = 1.
            # Fill output model
            model[i1, i2] = num/den

    return model


cpdef csibsons(int n1, int n2, float dh, np.ndarray[DTYPE_f, ndim=1] xp, np.ndarray[DTYPE_f, ndim=1] zp, np.ndarray[DTYPE_f, ndim=1] val):
    """
    Sibsons 2D interpolation.

    :param n1: number of points in the first dimension of the output model
    :param n2: number of points in the second dimension of the output model
    :param dh: spatial sampling
    :param xp: x-coordinates of points to interpolate
    :param zp: z-coordinates of points to interpolate
    :param val: value of the points to interpolate
    """

    cdef Py_ssize_t i1a, i2a, i1b, i2b, ipts, i1min, i1max, i2min, i2max

    # Get the number of points
    cdef int npts = np.size(xp)

    # Declare output array
    cdef np.ndarray[DTYPE_f, ndim=2] model = np.zeros((n1, n2), dtype=np.float32)

    # Declare variables
    cdef float xa, za, xb, zb, d, dmin
    cdef int ir

    # Declare arrays
    cdef np.ndarray[DTYPE_f, ndim=2] cap = np.zeros((n1, n2), dtype=np.float32)
    cdef np.ndarray[DTYPE_f, ndim=2] nap = np.zeros((n1, n2), dtype=np.float32)

    # First step: Voronoi interpolation
    cdef np.ndarray[DTYPE_f, ndim=2] vrn = cvoronoi(n1, n2, dh, xp, zp, val)

    # Loop over dimensions
    for i2a in range(0, n2):
        xa = float(i2a)*dh
        for i1a in range(0, n1):
            za = float(i1a)*dh
            # Initialize minimum distance
            dmin = 0.
            # Loop over number of scatter points
            for ipts in range(0, npts):
                # Calculate distances
                d = sqrt((xa-xp[ipts])**2+(za-zp[ipts])**2)
                if ipts == 0:
                    dmin = d
                    imin = ipts
                else:
                    if d < dmin:
                      dmin = d
                      imin = ipts
            # Calculate radius
            ir = int(dmin/dh)+1
            # Calculate application area
            i1min = 0
            i1max = n1-1
            i2min = 0
            i2max = n2-1
            if i1a-ir >= 0:
                i1min = i1a-ir
            if i1a+ir < n1:
                i1max = i1a+ir
            if i2a-ir >= 0:
                i2min = i2a-ir
            if i2a+ir < n2:
                i2max = i2a+ir
            # Loop over application area
            for i2b in range(i2min, i2max+1):
                xb = float(i2b)*dh
                for i1b in range(i1min, i1max+1):
                    zb = float(i1b)*dh
                    # Calculate distance
                    d = sqrt((xa-xb)**2+(za-zb)**2)
                    if d <= dmin:
                        cap[i1b, i2b] = cap[i1b, i2b]+vrn[i1a, i2a]
                        nap[i1b, i2b] = nap[i1b, i2b]+1.

    # Fill output model
    for i2a in range(0, n2):
        for i1a in range(0, n1):
            model[i1a, i2a] = cap[i1a, i2a]/nap[i1a, i2a]

    return model
