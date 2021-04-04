import os.path
import zlib


def lsb_range(lsb):
    "Check whether the LSB value is within the [1, 8] range."""
    if lsb <= 0 or lsb > 8:
        raise ValueError(f'LSB value must be within [1, 8], but got {lsb}')

def file_format(fname):
    """Check a filename does not have the JPEG extension."""
    _, ext = os.path.splitext(fname)
    if ext[1:].lower() in ('jpg', 'jpeg'):
        raise ValueError(f'Output file must not be in JPEG format')

def space_capacity(need, have):
    """Check there are enough pixels in the cover image for embedding."""
    if need > have:
        msg = f'Not enough space for embedding: {need:,}/{have:,}\n'
        msg += 'Either increase LSB embedding or use a bigger cover image.'
        raise ValueError(msg)

def data_integrity(data, crc):
    """Check the CRC-32 value is the same to that from the extracted data."""
    if zlib.crc32(data) != crc:
        raise ValueError('Data integrity not verified.')
