import os.path

import numpy as np
from PIL import Image

import utils.io as uio


def unpackbits(bytestream):
    return np.unpackbits(np.array(list(bytestream), dtype=np.uint8))

def packbits(bits):
    bits = bits.reshape((bits.shape[0]//8, 8))
    return bytes(np.packbits(bits))

def embed(cover_file, secret_file, stego_file):
    cover = uio.imread(cover_file, 'L')
    secret = uio.fread(secret_file)
    length_bits = unpackbits(len(secret).to_bytes(4, 'big'))
    ext = secret_file.split('.')[-1].ljust(4, '\x00').encode()
    ext_bits = unpackbits(ext)
    secret_bits = unpackbits(secret)
    bitstream = np.concatenate([length_bits, ext_bits, secret_bits])
    length = len(bitstream)
    stego = cover.flatten()
    stego[:length] = (stego[:length] & 0xfe) | bitstream
    stego = stego.reshape(cover.shape)
    uio.imsave(stego, stego_file)

def extract(stego_file):
    stego = uio.imread(stego_file).flatten()
    length_bits = stego[:32] & 0x01
    length = packbits(length_bits)
    length = sum(byte << 8*i for i, byte in enumerate(length[::-1])) * 8
    ext_bits = stego[32:64] & 0x01
    ext = packbits(ext_bits).decode().strip('\x00')
    secret_bits = stego[64:64+length] & 0x01
    secret = packbits(secret_bits)
    directory = os.path.dirname(stego_file)
    uio.fsave(secret, os.path.join(directory, f'extracted.{ext}'))


if __name__ == '__main__':
    folder = 'data/basic'
    cover_file = 'data/lena.png'
    secret_file = f'{folder}/lorem_ipsum.txt'
    stego_file = f'{folder}/lena_stego.png'
    embed(cover_file, secret_file, stego_file)
    extract(stego_file)
