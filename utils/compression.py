import zlib


def deflate(data):
    """Compress a bytestream with the Deflate algorithm."""
    compress = zlib.compressobj(level=9, wbits=-zlib.MAX_WBITS)
    deflated = compress.compress(data)
    deflated += compress.flush()
    return deflated

def inflate(data):
    """Uncompress a bytestream compressed with the Deflate algorithm."""
    decompress = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
    inflated = decompress.decompress(data)
    inflated += decompress.flush()
    return inflated
