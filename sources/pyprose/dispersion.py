import numpy as np

def compute_diagram(gobs: np.ndarray, w: np.ndarray, v: np.ndarray, offset: np.ndarray) -> np.ndarray:
    # Ensure correct shapes
    gobs = np.asarray(gobs, dtype=np.complex64).reshape(w.size, -1)  # Ensure shape (nw, noffset)
    w = np.asarray(w, dtype=np.float32).reshape(-1)  # Ensure shape (nw,)
    v = np.asarray(v, dtype=np.float32).reshape(-1)  # Ensure shape (nv,)
    offset = np.asarray(offset, dtype=np.float32).reshape(-1)  # Ensure shape (noffset,)

    ci = 1j  # Complex unit
    nv, nw, noffset = v.size, w.size, offset.size

    # Broadcasting to compute phase shift matrix
    phs = (w[:, None] * offset[None, :]) / v[:, None, None]  # Shape: (nv, nw, noffset)

    # Compute exponentials
    exp_terms = np.exp(ci * phs)  # Shape: (nv, nw, noffset)

    # Perform summation over `ir` dimension (axis=2) using broadcasting
    temp = np.sum(gobs.T * exp_terms, axis=2)  # Shape: (nv, nw)

    # Compute absolute values
    disp = np.abs(temp)

    return disp
