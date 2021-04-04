import os.path

import numpy as np

from utils.algorithms import optimal_pixel_adj
from utils.binary import bits2bytes, bytes2bits
from utils.compression import deflate, inflate
from utils.header import MAX_LENGTH, decode, encode
from utils.indices import permute_indices
import utils.io as uio
from utils.stats import psnr
import utils.validation as val


COLOR_PLANE = 2

def embed(cover_file, secret_file, out_file, lsb=1, passwd='', compress=True):
    """
    Embed a secret to an image with the pixel LSB substitution algorithm.

    Parameters
    ----------
    cover_file : str
        Path to cover image. If this a color image, the secret will be embedded
        in the B color channel.
    secret_file : str
        Path to secret file.
    out_file : str
        Destination for stego file. It must not be JPEG format as it is
        incompatible with the algorithm.
    lsb : int, optional
        Number of LSBs to embed the secret in. It must be in the range [1-8].
        Higher values introduce more distortion. Default is 1.
    passwd : str, optional
        Password to randomise the pixels where the secret will be embedded. This
        can be used to thwart sequential embedding steganalysis. If not set, the
        secret will be embedded sequentially. Using a password makes the code
        depend on numpy's PRNG and can't easily be ported to another language
        without implementating that feature exactly. Default is empty string.
    compress : bool, optional
        Compress the secret using gzip before embedding. If the compressed data
        is larger than the original secret, the algorithm will default to no
        compression even if the argument was set to True. Default is True.

    Returns
    -------
        None

    Notes
    -----
    The algorithm also implements the Optimal Pixel Adjustment method (OPA) to
    reduce the impact of the LSB substitution.
    """
    val.lsb_range(lsb)
    val.file_format(out_file)
    
    cover = uio.imread(cover_file)
    secret = uio.fread(secret_file)
    if compress:
        temp = deflate(secret)
        if len(temp) < len(secret):
            secret = temp
        else:
            compress = False

    header = encode(secret, secret_file, lsb, compress)
    stream = bytes2bits(secret, lsb)
    header_len = len(header)
    pixels_needed = header_len + len(stream)
    pixels_have = np.prod(cover.shape[:2])
    val.space_capacity(pixels_needed, pixels_have)
    print(f'{pixels_needed}/{pixels_have} pixels used')

    cover_plane = cover if cover.ndim < 3 else cover[...,COLOR_PLANE]
    stego_plane = cover_plane.copy()
    idx = permute_indices(cover_plane.shape, passwd=passwd, length=pixels_needed)

    header_mask = 0xfe
    header_idx = (idx[0][:header_len], idx[1][:header_len])
    stego_plane[header_idx] = (stego_plane[header_idx] & header_mask) | header
    idx = (idx[0][header_len:], idx[1][header_len:])
    if lsb == 1:
        stego_plane[idx] = (stego_plane[idx] & header_mask) | stream
    else:
        stream_mask = 256 - 2**lsb
        stego_plane[idx] = (stego_plane[idx] & stream_mask) | stream
        stego_plane[idx] = optimal_pixel_adj(stego_plane[idx], cover_plane[idx], lsb)
    print(f'PSNR = {psnr(cover_plane, stego_plane):2.2f}')

    if cover.ndim < 3:
        cover = stego_plane
    else:
        cover[...,COLOR_PLANE] = stego_plane
    uio.imsave(cover, out_file)
    
def extract(stego_file, passwd='', extraction_dir=''):
    """
    Extract a secret embedded with the pixel LSB substitution algorithm.

    Parameters
    ----------
    stego_file : str
        Path to the stego file.
    passwd : str, optional
        Password that was used during embedding. If this does not match, the
        program will exhibit undefined behaviour and will certainly crash,
        effectively failing to extract the secret. Default is empty string.
    extraction_dir : str, optional
        Directory where the secret will be extracted.

    Returns
    -------
    None

    See also
    --------
    embed : Embed secret. What this function reverses.
    """
    stego = uio.imread(stego_file)
    stego_plane = stego if stego.ndim < 3 else stego[...,COLOR_PLANE]
    idx = permute_indices(stego_plane.shape, passwd=passwd)
    
    header_idx = (idx[0][:MAX_LENGTH], idx[1][:MAX_LENGTH])
    bits = stego_plane[header_idx] & 0x01
    header = decode(bits)
    header_len = header['header_len']
    lsb = header['lsb']
    
    mask = 2**lsb - 1
    data_len = header['data_len']
    bitlength = int(np.ceil(data_len * 8 / lsb))
    idx = (idx[0][header_len:header_len+bitlength], idx[1][header_len:header_len+bitlength])
    stream = stego_plane[idx] & mask
    secret = bits2bytes(stream, lsb)[:data_len]
    val.data_integrity(secret, header['crc'])
    if header['compress']:
        secret = inflate(secret)
    directory = os.path.dirname(stego_file)
    out_file = os.path.join(directory, f'[extracted]{header["fname"]}')
    uio.fsave(secret, out_file)
    print(f'Secret extracted to "{out_file}"')


if __name__ == '__main__':
    folder = 'data/'
    passwd = 'hunter2'
    cover_file = f'{folder}lena.png'
    # this can be any file, including a text file for pure text
    secret_file = f'{folder}peppers.png'
    stego_file = f'{folder}lena_stego.png'

    embed(cover_file, secret_file, stego_file, lsb=2, passwd=passwd)
    extract(stego_file, passwd, folder)
