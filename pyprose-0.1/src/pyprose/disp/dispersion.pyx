import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float32_t DTYPE_f
ctypedef np.complex64_t DTYPE_c

@cython.boundscheck(False)
@cython.wraparound(False)

cpdef compute_diagram(np.ndarray[DTYPE_c, ndim=2] gobs, np.ndarray[DTYPE_f, ndim=1] w, np.ndarray[DTYPE_f, ndim=1] v, np.ndarray[DTYPE_f, ndim=1] offset):
    # FFT
    cdef Py_ssize_t iw, iv, ir
    cdef Py_ssize_t nw = w.size
    cdef Py_ssize_t nv = v.size
    cdef Py_ssize_t noffset = offset.size

    cdef DTYPE_c ci = complex(0., 1.)
    cdef DTYPE_f phs = 0.

    cdef np.ndarray[DTYPE_f, ndim=2] disp = np.zeros((nv, nw), dtype=np.float32)
    cdef np.ndarray[DTYPE_c, ndim=1] temp = np.zeros(nw, dtype=np.complex64)

    for iv in range(nv):
        temp[:] = complex(0., 0.)
        for ir in range(noffset):
            for iw in range(nw):
                phs = w[iw]*offset[ir]/v[iv]
                temp[iw] = temp[iw] + gobs[iw][ir]*np.exp(ci*phs)
        for iw in range(0, nw):
            disp[iv, iw] = disp[iv, iw] + np.abs(temp[iw])

    return disp