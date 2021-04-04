# LSB Steganography

This a demonstration of how to efficiently implement LSB steganography on Python. In general, the most frequently mistakes observed are

- Converting integers to strings for "bit manipulation" and then back to integers, when bitwise operations are more efficient
- Loading a pixel array instead of using the original (and most likely compressed) file bytestream
- Using loops instead of vectorisation

## lsb_basic.py

This is a stripped down version with no bells and whistles and with only embedding in 1 LSB for the simplest implementation.

## lsb_substitution.py

This is the main script, which supports the embedding in a dynamic number of least significant bits, compression of the secret, and the option to randomise the sequence of embedding pixels. It has a modular design so that it's easier build on top of it, or resuse various functions for different algorithms, e.g., embedding in DCT/DWT coefficients.