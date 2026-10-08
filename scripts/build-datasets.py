#!/usr/bin/env python3
"""Pack image datasets into sprite sheets the browser can decode quickly.

Usage:
    python3 scripts/build-datasets.py mnist   <dir with the four MNIST .gz files>        [out dir]
    python3 scripts/build-datasets.py fashion <dir with the four Fashion-MNIST .gz files> [out dir]
    python3 scripts/build-datasets.py cifar10 <path to cifar-10-binary.tar.gz>           [out dir]

Sources: MNIST (yann.lecun.com mirror), Fashion-MNIST (github.com/zalandoresearch/fashion-mnist,
same IDX format), CIFAR-10 binary version (www.cs.toronto.edu/~kriz/cifar.html).

Output (default: public/data), with <id> = mnist | fashion | cifar10:
    <id>-train-<k>.<ext>  5,000 training images per sheet
    <id>-test.<ext>       the test images
    <id>-labels.txt       one line of digits: training labels, then test labels

Sheets hold 100 images per row. Greyscale sets are lossless 8-bit PNG; CIFAR-10 is RGB JPEG
(quality 92, no chroma subsampling) to keep the download small. Requires Pillow.
"""
import gzip
import io
import os
import struct
import sys
import tarfile

from PIL import Image

SETS = {
    # id: (train count, test count, side, channels, extension)
    "mnist": (20_000, 2_000, 28, 1, "png"),
    "fashion": (10_000, 2_000, 28, 1, "png"),
    "cifar10": (10_000, 2_000, 32, 3, "jpg"),
}
CHUNK = 5_000
COLS = 100


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = struct.unpack(">I", data[:4])[0] & 0xFF
    return data[4 + 4 * ndim :]


def load_idx_dir(src):
    train_x = read_idx(os.path.join(src, "train-images-idx3-ubyte.gz"))
    train_y = read_idx(os.path.join(src, "train-labels-idx1-ubyte.gz"))
    test_x = read_idx(os.path.join(src, "t10k-images-idx3-ubyte.gz"))
    test_y = read_idx(os.path.join(src, "t10k-labels-idx1-ubyte.gz"))
    return train_x, train_y, test_x, test_y


def load_cifar(tar_path):
    """Returns images as interleaved RGB (32×32×3, row-major) and labels."""
    train_x, train_y, test_x, test_y = bytearray(), bytearray(), bytearray(), bytearray()
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {os.path.basename(m.name): m for m in tar.getmembers()}
        for name, xs, ys in [(f"data_batch_{i}.bin", train_x, train_y) for i in range(1, 6)] + [("test_batch.bin", test_x, test_y)]:
            raw = tar.extractfile(members[name]).read()
            for off in range(0, len(raw), 3073):
                ys.append(raw[off])
                planes = raw[off + 1 : off + 3073]
                r, g, b = planes[:1024], planes[1024:2048], planes[2048:]
                px = bytearray(3072)
                px[0::3], px[1::3], px[2::3] = r, g, b
                xs.extend(px)
    return bytes(train_x), bytes(train_y), bytes(test_x), bytes(test_y)


def sheet(images, start, count, side, channels):
    rows = (count + COLS - 1) // COLS
    mode = "L" if channels == 1 else "RGB"
    img = Image.new(mode, (COLS * side, rows * side))
    size = side * side * channels
    for i in range(count):
        tile = Image.frombytes(mode, (side, side), images[(start + i) * size : (start + i + 1) * size])
        img.paste(tile, ((i % COLS) * side, (i // COLS) * side))
    return img


def save(img, path, ext):
    if ext == "png":
        img.save(path, optimize=True)
    else:
        img.save(path, quality=92, subsampling=0, optimize=True)


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in SETS:
        sys.exit(__doc__)
    ds, src = sys.argv[1], sys.argv[2]
    dst = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "public", "data")
    os.makedirs(dst, exist_ok=True)
    n_train, n_test, side, channels, ext = SETS[ds]
    train_x, train_y, test_x, test_y = load_cifar(src) if ds == "cifar10" else load_idx_dir(src)

    for c in range(n_train // CHUNK):
        save(sheet(train_x, c * CHUNK, CHUNK, side, channels), os.path.join(dst, f"{ds}-train-{c}.{ext}"), ext)
    save(sheet(test_x, 0, n_test, side, channels), os.path.join(dst, f"{ds}-test.{ext}"), ext)
    with open(os.path.join(dst, f"{ds}-labels.txt"), "w") as f:
        f.write("".join(str(y) for y in train_y[:n_train] + test_y[:n_test]) + "\n")
    print(f"wrote {n_train} train + {n_test} test {ds} images to {os.path.abspath(dst)}")


if __name__ == "__main__":
    main()
