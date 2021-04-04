import numpy as np
from PIL import Image


def fread(fname):
    """Read bytestream from file."""
    with open(fname, 'rb') as f:
        data = f.read()
    return data

def fsave(bytestream, fname):
    """Save bytestream to file."""
    with open(fname, 'wb') as f:
        f.write(bytestream)
    
def imread(fname, mode=None):
    """Read pixel array from file."""
    img = Image.open(fname)
    if mode:
        img = img.convert(mode)
    return np.array(img)

def imsave(array, fname):
    """Save pixel array to file."""
    img = Image.fromarray(array)
    img.save(fname)
