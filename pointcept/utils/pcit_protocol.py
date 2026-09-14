"""
Wire protocol for the point cloud interpretability viewer ("pcit-arrays/1").

The viewer POSTs a whole scene as one binary body and expects the same framing
back. Nothing here depends on torch, so it can be imported and tested on its
own.

    [8]  uint64 little-endian   length of the JSON header
    [H]  UTF-8 JSON header
    [..] raw array bytes, back to back, each starting on an 8-byte boundary

Because the arrays are laid out contiguously and described by a numpy dtype
string, decoding is a `np.frombuffer(...).reshape(...)` per array -- no copy and
no per-point parsing, which is what keeps a 400k-point scene a single ~20 MiB
request instead of a JSON document.

Author: written for the pc-seg-interpretability viewer.
"""

import json
import struct

import numpy as np

FORMAT = "pcit-arrays/1"

# The dtypes the viewer emits. Anything else is a bug on one side or the other,
# and it is better to say so than to silently misread the body.
ALLOWED_DTYPES = {"|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8"}


def decode(payload):
    """
    Unpacks a request body.

    Returns (header, arrays) where `arrays` maps a name to an ndarray. The arrays
    are read-only views over `payload`; copy one before writing to it.
    """
    if len(payload) < 8:
        raise ValueError("payload is too short to hold a header length")
    (header_len,) = struct.unpack_from("<Q", payload, 0)
    if 8 + header_len > len(payload):
        raise ValueError("header length runs past the end of the payload")

    header = json.loads(bytes(payload[8:8 + header_len]).decode("utf-8"))
    base = 8 + header_len

    arrays = {}
    for spec in header.get("arrays", []):
        dtype = spec["dtype"]
        if dtype not in ALLOWED_DTYPES:
            raise ValueError(f"array {spec['name']!r} has unsupported dtype {dtype!r}")
        shape = tuple(spec["shape"])
        count = int(np.prod(shape)) if shape else 0
        offset = base + spec["offset"]
        end = offset + count * np.dtype(dtype).itemsize
        if end > len(payload):
            raise ValueError(f"array {spec['name']!r} runs past the end of the payload")
        arrays[spec["name"]] = np.frombuffer(
            payload, dtype=np.dtype(dtype), count=count, offset=offset
        ).reshape(shape)
    return header, arrays


def encode(arrays, **meta):
    """
    Packs a reply. `arrays` maps a name to anything array-like.

    Keyword arguments are merged into the JSON header, which is where
    `class_names`, `classes` and `num_points` belong.
    """
    specs, blobs, offset = [], [], 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        raw = arr.tobytes()
        specs.append({
            "name": name,
            "dtype": arr.dtype.str,
            "shape": list(arr.shape),
            "offset": offset,
            "nbytes": len(raw),
        })
        blobs.append(raw)
        # Pad so the next array starts on an 8-byte boundary, which is what lets
        # the other side take zero-copy views.
        padded = (offset + len(raw) + 7) & ~7
        blobs.append(b"\x00" * (padded - (offset + len(raw))))
        offset = padded

    header = json.dumps({"format": FORMAT, **meta, "arrays": specs}).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + b"".join(blobs)
