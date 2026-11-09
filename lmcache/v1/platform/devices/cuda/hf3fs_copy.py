# SPDX-License-Identifier: Apache-2.0
"""CUDA host registration and unordered scatter with explicit stream ownership."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import ctypes as C

_runtime = None


class Location(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class Attributes(C.Structure):
    _fields_ = [
        ("order", C.c_int),
        ("src", Location),
        ("dst", Location),
        ("flags", C.c_uint),
    ]


def runtime() -> Any:
    """Resolve the already-loaded CUDA runtime without initializing a GPU context."""
    global _runtime
    if _runtime is None:
        paths = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libcudart.so" in line
        }
        if len(paths) != 1:
            raise RuntimeError(f"Expected one loaded CUDA runtime, got {paths}")
        lib = C.CDLL(paths.pop())
        lib.cudaGetErrorString.restype = C.c_char_p
        lib.cudaHostRegister.argtypes = [C.c_void_p, C.c_size_t, C.c_uint]
        lib.cudaHostUnregister.argtypes = [C.c_void_p]
        lib.cudaMemcpyBatchAsync.argtypes = [
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_size_t),
            C.c_size_t,
            C.POINTER(Attributes),
            C.POINTER(C.c_size_t),
            C.c_size_t,
            C.c_void_p,
        ]
        lib.cudaMemcpyAsync.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.c_size_t,
            C.c_int,
            C.c_void_p,
        ]
        _runtime = lib
    return _runtime


def check(code: int) -> None:
    """Raise with the CUDA runtime error text when an operation fails."""
    if code:
        raise RuntimeError(
            f"CUDA {code}: {runtime().cudaGetErrorString(code).decode()}"
        )


def address(buffer: Any) -> int:
    """Return the base address of a writable contiguous host buffer."""
    return C.addressof(C.c_ubyte.from_buffer(buffer))


def register(buffer: Any) -> int:
    """Pin the host range for CUDA copies; the owner retains the mapping."""
    pointer = address(buffer)
    check(runtime().cudaHostRegister(pointer, len(buffer), 0))
    return pointer


def unregister(pointer: int) -> None:
    """Unpin a host range after every consuming CUDA operation has completed."""
    check(runtime().cudaHostUnregister(pointer))


def scatter(segments: Any, stream: int, overlap: bool = False) -> int:
    """Submit validated (dst, src, n) segments; caller retains sources until sync."""
    if not stream:
        raise ValueError("Batch copy requires a non-default stream")
    lib = runtime()
    if overlap:
        for dst, src, n in segments:
            check(lib.cudaMemcpyAsync(dst, src, n, 1, stream))
        return len(segments)
    n = len(segments)
    if not n:
        return 0
    dsts = (C.c_void_p * n)(*(d for d, _, _ in segments))
    srcs = (C.c_void_p * n)(*(s for _, s, _ in segments))
    sizes = (C.c_size_t * n)(*(z for _, _, z in segments))
    attrs, index = Attributes(1, Location(0, 0), Location(0, 0), 0), C.c_size_t(0)
    check(
        lib.cudaMemcpyBatchAsync(
            dsts, srcs, sizes, n, C.byref(attrs), C.byref(index), 1, stream
        )
    )
    return 1
