# SPDX-License-Identifier: Apache-2.0
"""Immutable, append-only FDB slot index; never reuses an allocated address.

No eviction/deletion. Pending allocations are invisible, and abandoned slots
stay allocated. Concurrent duplicate writers get distinct locations; the first
committed mapping wins. Transactions are idempotent after ambiguous commits.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any
import hashlib
import json
import struct
import threading
import uuid

# Third Party
import fdb
import msgpack

fdb.api_version(730)

PACK_BYTES = 229376
SLOT_BYTES = 262144
FILE_BYTES = 4 * 1024**3
SLOTS_PER_FILE = FILE_BYTES // SLOT_BYTES
LANES = 32


def coordinate(number: int) -> tuple[int, int]:
    """Map an append-only allocation number to a striped file and slot."""
    lane = number % LANES
    n = number // LANES
    return lane + (n // SLOTS_PER_FILE) * LANES, n % SLOTS_PER_FILE


def native_key(number: int) -> str:
    """Encode a committed slot coordinate for the native offset connector."""
    shard, slot = coordinate(number)
    return f"SLOT1@{shard:016x}@{slot:08x}"


def unpack(value: Any) -> Any:
    """Decode a complete pack after validating its declared structure and hashes."""
    return None if value is None else struct.unpack(">Q", value)[0]


class Index:
    def __init__(
        self,
        root: str,
        cluster_file: str,
        mount_point: str,
        cache_entries: int = 131072,
    ) -> None:
        self.root = str(Path(root).resolve())
        if not Path(self.root).is_relative_to(Path(mount_point).resolve()):
            raise ValueError("Index root must be under the configured 3FS mount")
        self.prefix = (
            b"LMCACHE-SLOTS-V1\x00"
            + hashlib.sha256(self.root.encode()).digest()
            + b"\x00"
        )
        self.db = fdb.open(cluster_file)
        self.db.options.set_transaction_timeout(10000)
        self.db.options.set_transaction_retry_limit(20)
        self.cache: OrderedDict[str, int] = OrderedDict()
        self.cache_entries = cache_entries
        self.lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        config = self.db[self.prefix + b"config"]
        if config is None:
            raise RuntimeError("Slot namespace not provisioned")
        self.config = json.loads(bytes(config))
        manifest = json.loads((Path(self.root) / ".slots-v1/layout.json").read_text())
        if manifest != self.config:
            raise RuntimeError("Filesystem/index generation mismatch")
        expected = {
            "root": self.root,
            "version": 1,
            "pack_bytes": PACK_BYTES,
            "slot_bytes": SLOT_BYTES,
            "file_bytes": FILE_BYTES,
            "lanes": LANES,
        }
        if any(self.config[k] != v for k, v in expected.items()):
            raise ValueError("Unsupported slot layout")
        self.capacity = self.config["files"] * SLOTS_PER_FILE

    def key(self, key: str) -> Any:
        """Encode an immutable identity for the corresponding persistence index."""
        if not isinstance(key, str) or not 1 <= len(key) <= 1024:
            raise ValueError("Invalid object key")
        return self.prefix + b"key/" + hashlib.sha256(key.encode()).digest()

    def remember(self, keys: list[str], values: Any) -> None:
        """Cache only committed positive coordinates within the metadata capacity."""
        with self.lock:
            for key, value in zip(keys, values, strict=True):
                if value is not None:
                    self.cache[key] = value
                    self.cache.move_to_end(key)
            while len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)

    def lookup(self, keys: list[str]) -> Any:
        """Resolve committed slots, preserving misses and request order."""
        with self.lock:
            values = [self.cache.get(k) for k in keys]
            self.counts["index_cache_hits"] += sum(v is not None for v in values)
        missing = [i for i, v in enumerate(values) if v is None]
        if missing:
            fetched = self._lookup(self.db, [self.key(keys[i]) for i in missing])
            for i, v in zip(missing, fetched, strict=True):
                values[i] = v
            self.remember(keys, values)
            with self.lock:
                self.counts["index_read_transactions"] += 1
        return values

    @fdb.transactional
    def _lookup(self, tr: Any, keys: list[str]) -> Any:
        futures = [tr.snapshot[k] for k in keys]
        return [unpack(v.value) for v in futures]

    def reserve(self, keys: list[str]) -> Any:
        """Allocate unique non-reusable slots while leaving new keys invisible to
        readers.
        """
        if not keys or len(keys) > 256 or len(set(keys)) != len(keys):
            raise ValueError("Reservation requires 1..256 distinct keys")
        ticket = uuid.uuid4().bytes
        values, write = self._reserve(self.db, [self.key(k) for k in keys], ticket)
        with self.lock:
            self.counts["index_reserve_transactions"] += 1
        return ticket, values, write

    @fdb.transactional
    def _reserve(self, tr: Any, keys: list[str], ticket: Any) -> Any:
        reservation = self.prefix + b"pending/" + ticket
        old = tr[reservation].value
        if old is not None:
            saved = msgpack.unpackb(bytes(old), raw=False)
            if saved[0] != keys:
                raise RuntimeError("Reservation identity mismatch")
            return saved[1:]
        futures = [tr[k] for k in keys]
        values = [unpack(v.value) for v in futures]
        write = [v is None for v in values]
        counter = self.prefix + b"next"
        n = unpack(tr[counter].value) or 0
        if n + sum(write) > self.capacity:
            raise RuntimeError("Slot namespace full")
        for i, needed in enumerate(write):
            if needed:
                values[i] = n
                n += 1
        tr[counter] = struct.pack(">Q", n)
        tr[reservation] = msgpack.packb([keys, values, write], use_bin_type=True)
        return values, write

    def commit(self, keys: list[str], ticket: Any, values: Any) -> Any:
        """Publish completed writes idempotently; concurrent duplicate keys keep the
        first mapping.
        """
        result = self._commit(self.db, [self.key(k) for k in keys], ticket, values)
        self.remember(keys, result)
        with self.lock:
            self.counts["index_commit_transactions"] += 1
        return result

    @fdb.transactional
    def _commit(self, tr: Any, keys: list[str], ticket: Any, values: Any) -> Any:
        reservation = self.prefix + b"pending/" + ticket
        pending = tr[reservation].value
        futures = [tr[k] for k in keys]
        current = [unpack(v.value) for v in futures]
        if pending is None:
            # A retry after commit_unknown_result may find the already-published
            # mappings, including a different duplicate writer's winning slots.
            if any(v is None for v in current):
                raise RuntimeError("Missing reservation")
            return current
        saved = msgpack.unpackb(bytes(pending), raw=False)
        if saved[0] != keys or saved[1] != values:
            raise RuntimeError("Reservation mismatch")
        for i, (key, value) in enumerate(zip(keys, values, strict=True)):
            if current[i] is None:
                tr[key] = struct.pack(">Q", value)
                current[i] = value
        del tr[reservation]
        return current

    def abort(self, ticket: Any):
        # Never reuse these positions -> None: an interrupted writer may still own IO.
        """Forget a pending reservation without reclaiming its potentially in-flight
        addresses.
        """
        del self.db[self.prefix + b"pending/" + ticket]

    def stats(self) -> dict[str, Any]:
        """Snapshot bounded index cache occupancy and transaction counters."""
        with self.lock:
            return dict(
                self.counts,
                index_cache_entries=len(self.cache),
                index_cache_capacity=self.cache_entries,
            )
