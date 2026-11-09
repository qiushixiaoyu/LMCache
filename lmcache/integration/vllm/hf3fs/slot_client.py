# SPDX-License-Identifier: Apache-2.0
"""Native connector facade: FDB coordinates + existing asynchronous USRBIO.

Buffers stay strongly referenced until native completions have drained. Index
publication follows completed writes, and never substitutes for IO completion.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any
import os
import select
import threading
import time

# Third Party
import torch  # noqa: F401 -- initializes libtorch before the native extension

# First Party
from lmcache.lmcache_hf3fs import LMCacheHf3fsClient
from lmcache.logging import init_logger

# Local
from .slot_index import PACK_BYTES, Index, native_key

logger = init_logger(__name__)


class SlotClient:
    def __init__(self, cfg: dict[str, Any], **arena: Any):
        # A metadata cache the same size as the payload pool mostly duplicates
        # that pool -> None: its misses then query FDB again. Retain eight payload-pool
        # generations of cheap coordinates in the shared reader. No KV payload
        # or GPU memory capacity changes here.
        self.index = Index(
            cfg["base_paths"],
            cfg["cluster_file"],
            cfg["mount_point"],
            cache_entries=cfg.get(
                "index_cache_entries", 1048576 if "read_buffer" in arena else 131072
            ),
        )
        self.legacy_writes = cfg.get("legacy_writes", False)
        self.native = LMCacheHf3fsClient(
            cfg["mount_point"],
            cfg["base_paths"],
            num_workers=cfg.get("num_workers", 8),
            ior_entries=cfg.get("ior_entries", 128),
            io_depth=0,
            iov_size=8 * 1024**2,
            enable_key_buffer=False,
            per_op_workers={"store": cfg.get("store_workers", 32)},
            immutable_fd_capacity=cfg.get("immutable_fd_capacity", 8192),
            hash_shards=True,
            **arena,
        )
        self.fd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self.lock = threading.Lock()
        self.native_lock = threading.Lock()
        self.pending = {}
        self.completed = []
        self.sequence = 0
        self.closed = False
        self.counts = Counter()
        self.slots = threading.BoundedSemaphore(64)
        self.executor = ThreadPoolExecutor(12, thread_name_prefix="slot-index")
        self.stop = threading.Event()
        self.collector = threading.Thread(
            target=self._collect, name="slot-native-cqe", daemon=True
        )
        self.collector.start()

    def _collect(self) -> None:
        poller = select.poll()
        poller.register(self.native.event_fd(), select.POLLIN)
        while not self.stop.is_set():
            if not poller.poll(100):
                continue
            with self.native_lock:
                done = self.native.drain_completions()
                pairs = [
                    (self.pending.pop(int(fid)), (ok, error, bits))
                    for fid, ok, error, bits in done
                ]
            for future, result in pairs:
                future.set_result(result)

    def _io(self, op: str, keys: list[str], buffers: Any = None) -> Any:
        if not keys:
            return []
        future = Future()
        with self.native_lock:
            fn = getattr(self.native, "submit_batch_" + op)
            fid = fn(keys) if buffers is None else fn(keys, buffers)
            self.pending[int(fid)] = future
        # No timeout can authorize freeing a buffer with outstanding DMA.
        ok, error, bits = future.result()
        if op == "set" and not ok:
            raise RuntimeError("Slot native write failed: " + error)
        if bits is None:
            if not ok:
                raise RuntimeError(error)
            return [True] * len(keys)
        return list(bits)

    def _operate(self, op: str, keys: list[str], buffers: Any) -> Any:
        start = time.perf_counter()
        if op == "set":
            if self.legacy_writes:
                return self._io("set", keys, buffers)
            ticket, values, write = self.index.reserve(keys)
            try:
                todo = [i for i, v in enumerate(write) if v]
                self._io(
                    "set",
                    [native_key(values[i]) for i in todo],
                    [buffers[i] for i in todo],
                )
                self.index.commit(keys, ticket, values)
            except BaseException:
                # Do not erase a reservation on ambiguous FDB errors: keeping it
                # is safe, and permits inspection/recovery without reusing slots.
                raise
            bits = [True] * len(keys)
        else:
            values = self.index.lookup(keys)
            if op == "exists":
                bits = [v is not None for v in values]
                missing = [i for i, v in enumerate(values) if v is None]
                old = self._io("exists", [keys[i] for i in missing])
                for i, hit in zip(missing, old, strict=True):
                    bits[i] = hit
            else:
                translated = [
                    key if value is None else native_key(value)
                    for key, value in zip(keys, values, strict=True)
                ]
                bits = self._io("get", translated, buffers)
        with self.lock:
            self.counts[op + "_batches"] += 1
            self.counts[op + "_ns"] += int((time.perf_counter() - start) * 1e9)
        return bits

    def _submit(self, op: str, keys: list[str], buffers: Any = None) -> Any:
        keys = list(keys)
        if not 1 <= len(keys) <= 256:
            raise ValueError("Slot batch size must be 1..256")
        if buffers is not None:
            buffers = list(buffers)
            if len(buffers) != len(keys) or any(
                b.nbytes != PACK_BYTES for b in buffers
            ):
                raise ValueError("Only complete canonical packs supported")
        if not self.slots.acquire(blocking=False):
            raise RuntimeError("Slot IO admission full")
        with self.lock:
            if self.closed:
                self.slots.release()
                raise RuntimeError("Client closed")
            fid = self.sequence
            self.sequence += 1
            try:
                self.executor.submit(self._run, fid, op, keys, buffers)
            except BaseException:
                self.slots.release()
                raise
        return fid

    def _run(self, fid: Any, op: str, keys: list[str], buffers: Any) -> Any:
        try:
            result = (fid, True, "", self._operate(op, keys, buffers))
        except Exception as error:
            logger.exception("Canonical slot operation failed")
            result = (fid, False, str(error), [False] * len(keys))
        with self.lock:
            self.completed.append(result)
            os.eventfd_write(self.fd, 1)
        self.slots.release()

    def submit_batch_get(self, keys: list[str], buffers: Any) -> int:
        """Queue reads while retaining destination views until completion."""
        return self._submit("get", keys, buffers)

    def submit_batch_set(self, keys: list[str], buffers: Any) -> int:
        """Queue immutable writes; acknowledge only after I/O and index publication."""
        return self._submit("set", keys, buffers)

    def submit_batch_exists(self, keys: list[str]) -> int:
        """Queue committed presence lookup, including legacy-file fallback."""
        return self._submit("exists", keys)

    def event_fd(self) -> int:
        """Return the eventfd used by the adapter to observe completed tasks."""
        return self.fd

    def drain_completions(self) -> Any:
        """Drain task results without releasing buffers before native completion."""
        with self.lock:
            try:
                os.eventfd_read(self.fd)
            except BlockingIOError:
                pass
            result, self.completed = self.completed, []
            return result

    def metadata_stats(self) -> dict[str, Any]:
        """Snapshot metadata, registration and index counters."""
        with self.lock:
            counts = dict(self.counts)
        return dict(self.native.metadata_stats(), **self.index.stats(), **counts)

    def clear_fd_cache(self) -> None:
        """Retire cached handles; outstanding native leases retain their descriptors."""
        self.native.clear_fd_cache()

    def close(self) -> None:
        """Drain owned work before releasing its resources."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
        self.executor.shutdown(wait=True)
        assert not self.pending
        self.stop.set()
        self.collector.join()
        self.native.close()
        os.close(self.fd)
