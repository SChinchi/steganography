import numpy as np


# We work with little-endian order because we rely on `np.packbits` under the
# hood and bit arrays shorter than 8 bits are tail padded with zeroes. For
# example, 10011 (19 in dec), is interpreted as 10011000 and not 00010011.
# In little endian this works as intended, because 11001 becomes 11001000.
ORDER = 'little'

def bits2bytes(bits, group=1):
    """
    Combine bit groups to bytes.

    Parameters
    ----------
    bits : ndarray, uint8 type
         1-D array of bit groups. The bits are expected to be in little-endian
         order.
    group : int, optional
         The number of bits each value in `bits` represents. For group = 1, we
         will need 8 elements to form a byte. For group = 3, the first byte
         will be composed by the first two elements and two bits from the third
         one. If the last byte doesn't have enough for 8 bits, it will be
         padded with zeroes. Default is 1.

    Returns
    -------
    bytes object
        The resultant bytestream.

    See also
    --------
    bytes2bits : The opposite operation

    Examples
    --------
    >>> bits = np.array([0, 1, 1, 0, 0, 1, 1, 1], dtype=np.uint8)  # 230
    >>> bits2bytes(bits)
    b'\xe6'
    
    >>> bits = np.array([4, 3, 6], dtype=np.uint8) # 001, 110, 011 in little endian
    >>> bits2bytes(bits, 3) # little endian of 00111001, 10000000
    b'\x9c\x01'
    """
    length = len(bits) // group
    if group > 1:
        bits = np.unpackbits(bits[:,None], axis=1, bitorder=ORDER)
        bits = bits[:,:group].flatten()
    return bytes(np.packbits(bits, bitorder=ORDER))

def bytes2bits(bytestream, group=1):
    """
    Split a bytestream to bit groups.

    Parameters
    ----------
    bytestream : bytes
         Input bytestream.
    group : int, optional
         The number of bits to split per group.

    Returns
    -------
    ndarray, uint8 type
        Array with the resultant bit groups. The values will be in little-endian
        order.

    See also
    --------
    bits2bytes : The opposite operation

    Examples
    --------
    >>> value = bytes([176])
    >>> bytes2bits(value)
    array([0, 0, 0, 0, 1, 1, 0, 1], dtype=uint8)

    >>> value = bytes([176, 34]) # in little endian 00001101, 01000100
    >>> bytes2bits(value, 3) # 000, 011, 010, 100, 010, 000
    array([0, 6, 2, 1, 2, 0], dtype=uint8)
    """
    bits = np.unpackbits(np.array(list(bytestream), dtype=np.uint8), bitorder=ORDER)
    if group == 1:
        return bits
    pad = len(bits) % group
    if pad:
        bits = np.concatenate([bits, np.zeros((group-pad,), dtype=np.uint8)])
    bits = bits.reshape((bits.shape[0]//group, group))
    return np.squeeze(np.packbits(bits, axis=1, bitorder=ORDER))

def packbits(bits):
    """
    Combine a bitstream to a 32-bit unsigned integer.

    Parameters
    ----------
    bits : ndarray, uint8 type
        Array with the input bitstream in little-endian order. This cannot be
        more than 32 bits.

    Returns
    -------
    int, uint32 type
        Resultant integer.

    Examples
    --------
    >>> bits = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
    >>> packbits(bits)
    1452
    """
    if len(bits) > 32:
        raise ValueError(f'Too many bits. Excepted maximum 32, got {len(bits)}.')
    n = np.packbits(bits, bitorder=ORDER)
    if len(n) > 1:
        n = n << np.arange(0, 8*len(n), 8, dtype=np.uint32)
    return n.sum()

def unpackbits(n, length=None):
    """
    Split an integer to its little-endian bitstream.

    Parameters
    ----------
    n : int
        Input number, treated as 32-bit unsigned.
    length : int, optional
        The number of bits to show. If not defined, it will calculate the
        minimum number of bits required. Can be used to pad the bitstream
        to a desired length.

    Returns
    -------
    bits : ndarray, uint8 type
        Array with the resultant bitstream in little-endian order. The array
        will contain the minimum number of bits to express the input, i.e., it
        will not be padded with zeros to length 32.

    Examples
    --------
    >>> unpackbits(4527)
    array([1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1], dtype=uint8)
    """
    if length is None:
        length = int(np.floor(np.log2(n) + 1))
    bits = np.unpackbits(np.array([n], dtype=np.uint32).view(np.uint8), bitorder=ORDER)
    return bits[:length]
