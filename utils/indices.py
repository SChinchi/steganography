import hashlib

import numpy as np


def permute_indices(shape, passwd='', length=None):
    """
    Shuffle the indices of a 2D array.

    Parameters
    ----------
    shape : iterable
        A tuple-like object which describes the height and width of an image.
    passwd : str, optional
        A string used as a seed for shuffling the indices. If it is an empty
        string, no shuffling will be done and the indices will be returned in
        order. Default is empty string.
    length : None or int, optional
        The number of pixel coordinates to output. If not defined, all pixels
        will be returned. Default is None.

    Returns
    -------
    out : tuple of ndarrays
        A tuple containing the rows and the columns of the permuted image
        pixels. The format is to be directly used for numpy array indexing.

    Examples
    --------
    >>> a = np.array([
                [156, 234, 128, 202],
                [72, 125, 58, 111],
                [99, 198, 24, 201]],
            dtype=np.uint8)
    >>> idx = permute_indices(a.shape)
    >>> idx
    (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int32),
    array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int32))
    >>> a[idx]
    array([156, 234, 128, 202,  72, 125,  58, 111,  99, 198,  24, 201],
          dtype=uint8)
    >>> idx = permute_indices(a.shape, 'hello world', 7)
    >>> a[idx]
    array([202, 198, 201, 111,  72,  58, 125], dtype=uint8)
    """
    # Creating a 1D array for all indices, shuffling it and converting it to 2D
    # coordinates with div and mod is significantly faster than shuffling tuples
    # of coordinates from `itertools.product` and zipping that to (all x, all y)
    # for numpy indexing.
    idx = np.arange(np.prod(shape))
    length = length if length is not None else len(idx)
    if passwd != '':
        # We can't rely on `hash(passwd)` in Python 3, because it returns a
        # different value for each run
        seed = int(hashlib.sha256(passwd.encode()).hexdigest(), 16) & 0xffffffff
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    idx = idx[:length]
    return idx // shape[1], idx % shape[1]
