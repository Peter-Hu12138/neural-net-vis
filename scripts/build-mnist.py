#!/usr/bin/env python3
"""Pack a subset of MNIST into PNG sprite sheets for the browser.

Usage:
    python3 scripts/build-mnist.py <dir with the four original .gz files> [out dir]

The original files are the standard MNIST distribution
(train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz,
t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz).

Output (default: public/data):
    mnist-train-0.png ... mnist-train-3.png  5,000 training digits each
    mnist-test.png                            2,000 test digits
    mnist-labels.bin                          22,000 uint8 labels (train, then test)

Each sprite is a grayscale PNG with 100 digits per row, 28x28 px per digit,
white ink on black, exactly as stored in MNIST. Only the standard library is used.
"""
import gzip
import os
import struct
import sys
import zlib

TRAIN_COUNT = 20_000
TEST_COUNT = 2_000
CHUNK = 5_000
COLS = 100
SIDE = 28


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack(">I", data[:4])[0]
    ndim = magic & 0xFF
    dims = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
    return dims, data[4 + 4 * ndim :]


def write_png(path, width, height, pixels):
    """Write an 8-bit grayscale PNG. `pixels` is a bytes object, row-major."""

    def chunk(tag, payload):
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter type: none (zlib does well enough on MNIST)
        raw += pixels[y * width : (y + 1) * width]
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def sprite(images, start, count):
    rows = (count + COLS - 1) // COLS
    width, height = COLS * SIDE, rows * SIDE
    out = bytearray(width * height)
    for i in range(count):
        src = images[(start + i) * SIDE * SIDE : (start + i + 1) * SIDE * SIDE]
        ox, oy = (i % COLS) * SIDE, (i // COLS) * SIDE
        for r in range(SIDE):
            o = (oy + r) * width + ox
            out[o : o + SIDE] = src[r * SIDE : (r + 1) * SIDE]
    return width, height, bytes(out)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "public", "data")
    os.makedirs(dst, exist_ok=True)

    _, train_x = read_idx(os.path.join(src, "train-images-idx3-ubyte.gz"))
    _, train_y = read_idx(os.path.join(src, "train-labels-idx1-ubyte.gz"))
    _, test_x = read_idx(os.path.join(src, "t10k-images-idx3-ubyte.gz"))
    _, test_y = read_idx(os.path.join(src, "t10k-labels-idx1-ubyte.gz"))

    for c in range(TRAIN_COUNT // CHUNK):
        w, h, px = sprite(train_x, c * CHUNK, CHUNK)
        write_png(os.path.join(dst, f"mnist-train-{c}.png"), w, h, px)
    w, h, px = sprite(test_x, 0, TEST_COUNT)
    write_png(os.path.join(dst, "mnist-test.png"), w, h, px)

    with open(os.path.join(dst, "mnist-labels.bin"), "wb") as f:
        f.write(train_y[:TRAIN_COUNT] + test_y[:TEST_COUNT])

    print(f"wrote {TRAIN_COUNT} train + {TEST_COUNT} test digits to {os.path.abspath(dst)}")


if __name__ == "__main__":
    main()
