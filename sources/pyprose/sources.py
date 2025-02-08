import numpy as np

def ricker_wavelet(nt, dt, f0, t0):
    """
    Generate a Ricker wavelet (source time function).

    Parameters:
        nt  (int): Number of time samples
        dt  (float): Time sampling interval
        f0  (float): Peak frequency
        t0  (float): Delay

    Returns:
        t (numpy array): Time vector
        w (numpy array): Ricker wavelet values
    """
    t = np.linspace(0, (nt - 1) * dt, nt)  # Time vector
    t = t - t0  # Center around t0
    w = (1 - 2 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)
    return np.linspace(0, (nt - 1) * dt, nt), w