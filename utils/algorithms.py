import numpy as np


def optimal_pixel_adj(modified, original, k):
    """
    Minimise the value change after LSB substitution.

    LSB substitution minimises the Hamming distance, while OPA [1]_ minimises
    the phenotype distance. For example, assume a 000 substitution in the 3
    least significant bits:
    
        LSB : xxxx0111 -> xxxx0000 (difference of 7)
        OPA : xxxx0111 -> xxxx1000 (difference of 1)
        
    LSB substitution of N bits can result to a maximum change of (2**N)-1, while
    OPA only of 2**(N-1).

    Parameters
    ----------
    modified : ndarray
        Modified image pixels with LSB substitution of uint8 type.
    original : ndarray
        Original image pixels of uint8 type.
    k : int
        Value of least significant bits where embedding takes place.

    Returns
    -------
    out : ndarray, uint8 type
        The corrected value for each pixel within the [0, 255] value range.

    References
    ----------
    .. [1] Chan, C. K., & Cheng, L. M. (2004). Hiding data in images by simple
    LSB substitution. pattern recognition, 37(3), 469-474.

    Example
    -------
    >>> lsb = 3
    >>> mask = 255 - (2**lsb - 1)
    >>> p0 = np.array([157, 160], dtype=np.uint8)
    >>> stream = np.array([7, 7], dtype=np.uint8)
    >>> p1 = (p0 & mask) | stream
    >>> p1
    array([159, 167], dtype=uint8)
    >>> optimal_pixel_adj(p1, p0, lsb)
    array([159, 159], dtype=uint8)
    """
    opa = []
    inner = 2**(k-1)
    outer = 2**k
    for i in range(-outer+1, outer):
        if inner < i < outer:
            opa.append(-outer)
        elif -inner > i > -outer:
            opa.append(outer)
        else:
            opa.append(0)
    opa = np.array(opa)
    diff = modified.astype(np.int32) - original
    correction = opa[diff - outer]
    c = modified + correction
    correction[(c < 0) | (c > 255)] = 0
    out = modified + correction
    return out.astype(np.uint8)
