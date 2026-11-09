# SPDX-License-Identifier: Apache-2.0
"""Immutable eight-component packs: 256-byte header and 128-token chunks."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import hashlib
import json
import struct

# Local
from .canonical_protocol import validate

COMPONENTS = tuple(
    f"layer_{layer}/{kind}"
    for layer in (2, 8, 14, 20)
    for kind in ("main_kv", "indexer_k")
)
SIZES = tuple(2 * n for n in (18688, 4224, 18688, 4224, 18688, 4224, 37376, 8448))
LAYOUT = dict(zip(COMPONENTS, SIZES, strict=True))
LAYOUT_HASH = hashlib.sha256(json.dumps(list(LAYOUT.items())).encode()).digest()
HEADER = struct.Struct("!8s32s32s32sQ144x")
MAGIC = b"DSV41PK2"
BODY_SIZE = sum(SIZES)
PACK_SIZE = HEADER.size + BODY_SIZE
CHUNK_TOKENS = 128
MAX_PACKS = 256
assert PACK_SIZE == 229376 and PACK_SIZE % 4096 == 0


def split_key(key: str) -> tuple[str, str, str]:
    """Return the canonical namespace, component and full-prefix hash."""
    fmt, namespace, component, prefix = key.split("@")
    return fmt + "@" + namespace, component, prefix


def group_key(key: str) -> str:
    """Return the namespace and chained-prefix identity of a component key."""
    namespace, _, prefix = split_key(key)
    return namespace + "@" + prefix


def descriptor(group: str, component: str) -> list[Any]:
    """Encode a component request with its canonical byte size."""
    namespace, prefix = group.rsplit("@", 1)
    return [namespace + "@" + component + "@" + prefix, LAYOUT[component]]


def checked_groups(operation: str, rows: list[list[Any]]) -> Any:
    """Validate descriptors and group component rows by immutable pack identity."""
    validate(operation, rows)
    groups = {}
    for row in rows:
        key, size = row[:2]
        _, component, _ = split_key(key)
        if size != LAYOUT[component]:
            raise ValueError("Packed layout size mismatch")
        group = groups.setdefault(group_key(key), {})
        if component in group and group[component] != row:
            raise ValueError("Conflicting duplicate component")
        group[component] = row
    return groups


def pack(group: str, rows: list[list[Any]]) -> bytes:
    """Serialize all eight components and their versioned header into one complete
    pack.
    """
    components = checked_groups("put", rows)
    if set(components) != {group} or set(components[group]) != set(COMPONENTS):
        raise ValueError("A packed PUT requires all eight components")
    body = b"".join(components[group][c][2] for c in COMPONENTS)
    return (
        HEADER.pack(
            MAGIC,
            LAYOUT_HASH,
            hashlib.sha256(group.encode()).digest(),
            hashlib.sha256(body).digest(),
            len(body),
        )
        + body
    )


def unpack(group: str, data: Any, verify_hashes: bool = True) -> Any:
    """Decode a complete pack after validating its declared structure and hashes."""
    if len(data) != PACK_SIZE:
        raise ValueError("Short or oversized packed object")
    magic, layout, identity, digest, size = HEADER.unpack_from(data)
    body = memoryview(data)[HEADER.size :]
    if (
        magic != MAGIC
        or size != BODY_SIZE
        or (
            verify_hashes
            and (
                layout != LAYOUT_HASH
                or identity != hashlib.sha256(group.encode()).digest()
                or digest != hashlib.sha256(body).digest()
            )
        )
    ):
        raise ValueError("Packed header, identity or checksum mismatch")
    result = {}
    offset = 0
    for component, size in LAYOUT.items():
        result[component] = bytes(body[offset : offset + size])
        offset += size
    return result
