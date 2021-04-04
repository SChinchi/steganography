import os.path
import zlib

import numpy as np

from .binary import packbits, unpackbits


# Maximum header length: 2121 bits
# data_len: 5 + 32, lsb: 3, fname_len: 8, fname: 255 * 8, compress: 1, crc: 32
MAX_LENGTH = 2121

def _message_length_from_bits(bits):
    header_len = packbits(bits[:5])
    data_len = packbits(bits[5:5+header_len])
    return data_len, header_len + 5

def _message_length_to_bits(length, bits=32):
    length_binary = unpackbits(length)
    length_header = unpackbits(len(length_binary), 5)
    return np.concatenate([length_header, length_binary])

def decode(bits):
    """
    Decode the header info.

    Parameters
    ----------
    bits : ndarray, uint8 type
        1-D array of 1s and 0s.

    Returns
    -------
    out : dict
        Necessary information to extract the secret.
        - 'data_len': Bytestream length of the secret.
        - 'lsb': The number of LSBs it has been embedded into.
        - 'fname': The original filename of the secret.
        - 'compress': Whether the secret has been compressed before embedding.
                      In this case 'data_len' refers to the compressed length.
        - 'crc': CRC-32 value for validation. If the secret has been compressed,
                 the checksum value is calculated for the compressed bytestream.
    """ 
    data_len, index = _message_length_from_bits(bits)
    lsb = packbits(bits[index:index+3]) + 1
    index += 3
    fname_len = packbits(bits[index:index+8])
    index += 8
    fname = ''.join(map(chr, np.packbits(bits[index:index+8*fname_len], bitorder='little')))
    index += 8 * fname_len
    compress = bool(bits[index])
    index += 1
    crc = packbits(bits[index:index+32])
    index += 32
    return {'data_len': data_len, 'lsb': lsb, 'fname': fname,
            'compress': compress, 'crc': crc, 'header_len': index}
    
def encode(data, fname, lsb, compress):
    """
    Encode the necessary secret information for proper extraction later on.

    Parameters
    ----------
    data : bytes
        The bytestream of the secret.
    fname : str
        The filename of the secret.
    lsb : int
        The number of LSBs used for embedding.
    compress : bool
        Whether the secret has been compressed.

    Returns
    -------
    ndarray, uint8 type
        1-D array of 1s and 0s.
    """
    data_len = _message_length_to_bits(len(data))
    lsb_bin = unpackbits(lsb-1, 3)
    _, fname = os.path.split(fname)
    fname_len = unpackbits(len(fname), 8)
    fname_bin = np.unpackbits(np.array(list(fname.encode()), dtype=np.uint8), bitorder='little')
    compress_bin = np.array([bool(compress)], dtype=np.uint8)
    crc = unpackbits(zlib.crc32(data), 32)
    return np.concatenate([data_len, lsb_bin, fname_len, fname_bin, compress_bin, crc])
