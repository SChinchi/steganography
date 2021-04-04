import numpy as np


def mse(a, b):
    """Mean square error between a modified image and the original."""
    x, y = a.shape
    a = a.astype(np.int32)
    return np.sum((a - b)**2) / (x * y)

def psnr(a, b, a_max=255):
    """Peak signal-to-noise ratio between a modified image and the original."""
    e = mse(a, b)
    return 10 * np.log10(a_max**2 / e) if e else np.inf
