# SPDX-License-Identifier: Apache-2.0
"""Bounded immutable pack slots, singleflight reads and explicit consumer leases.

Consumers release leases explicitly after I/O and GPU completion. No process
liveness polling or automatic reclamation of abandoned leases is performed.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter, OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import threading
import time
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_open_pool_as_mmap,
    shm_unlink,
)

# Local
from .pack_format import HEADER, LAYOUT, PACK_SIZE, group_key, split_key

logger = init_logger(__name__)

OFFSETS = {}
offset = HEADER.size
for component, length in LAYOUT.items():
    OFFSETS[component] = offset
    offset += length


def process_cookie(pid: int) -> str | None:
    """Read a process-start cookie once for compatibility with existing consumers."""
    try:
        # comm may contain spaces and parentheses; fields after its closing ')'.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


@dataclass
class Entry:
    slot: int
    generation: int
    refs: int = 0
    touched: float = field(default_factory=time.monotonic)
    ready: Future = field(default_factory=Future)


class PackPool:
    def __init__(self, loader: Any, slots: int = 1024, ttl: float = 60) -> None:

        self.name = "lc3fs_read_" + uuid.uuid4().hex
        self.capacity = slots * PACK_SIZE
        self.address = shm_create_readwrite(self.name, self.capacity)
        self.mapping = shm_open_pool_as_mmap(self.name, self.capacity)
        self.loader, self.ttl = loader, ttl
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        self.failed = set()
        self.free = deque(range(slots))
        self.generations = [0] * slots
        self.leases = {}
        self.finished = OrderedDict()
        self.counts = Counter()
        self.active = 0
        self.closing = False
        self.executor = ThreadPoolExecutor(4, thread_name_prefix="shm-native")

    def _drop(self, group: str, reason: Any = None) -> None:
        e = self.entries[group]
        assert not e.refs and e.ready.done()
        self.free.append(e.slot)
        del self.entries[group]
        self.failed.discard(group)
        if reason:
            self.counts[reason] += 1

    def _finish(self, token: str) -> None:
        lease = self.leases.pop(token, None)
        if lease is not None:
            for e in lease["entries"].values():
                e.refs -= 1
                assert e.refs >= 0
        self.finished[token] = None
        while len(self.finished) > 4096:
            self.finished.popitem(last=False)

    def _reap(self) -> None:
        # Caller holds the pool lock. Keep cache expiry, omit all lease/PID scans.
        for group in list(self.failed):
            entry = self.entries[group]
            if not entry.refs and entry.ready.done():
                self._drop(group, "failed_packs_removed")
        now = time.monotonic()
        expired = []
        for group, entry in self.entries.items():
            self.counts["expiry_entries_examined"] += 1
            if now - entry.touched <= self.ttl:
                break
            if not entry.refs and entry.ready.done():
                expired.append(group)
        for group in expired:
            self._drop(group, "expired_packs")

    def _load(self, owned: Any) -> None:
        try:
            valid = self.loader([(g, e.slot) for g, e in owned])
            with self.lock:
                for g, e in owned:
                    e.ready.set_result(g in valid)
                    if g not in valid:
                        self.failed.add(g)
        except BaseException:
            with self.lock:
                self.counts["load_errors"] += 1
                # Store failure text, not traceback references holding tensors.
                for g, e in owned:
                    e.ready.set_result(False)
                    self.failed.add(g)
        finally:
            with self.lock:
                self.active -= 1

    def acquire(
        self,
        rows: list[list[Any]],
        token: str,
        pid: int,
        cookie: str | None,
        releases: Any = (),
    ) -> Any:
        """Return component offsets under a lease, waiting for missing packs outside
        the lock.
        """
        if (
            not isinstance(token, str)
            or len(token) != 32
            or any(c not in "0123456789abcdef" for c in token)
            or type(pid) is not int
            or pid <= 0
        ):
            raise ValueError("Invalid lease identity")
        # Do not read /proc on GET. The Unix socket checks its peer PID;
        # cookie remains opaque metadata for compatibility with existing clients.
        # Explicit leases and completed I/O, not PID lifetime, protect slots.
        # Parse keys and offsets before taking the shared state lock.
        plan = [
            (group_key(key), OFFSETS[split_key(key)[1]], length) for key, length in rows
        ]
        groups = list(dict.fromkeys(group for group, _, _ in plan))
        if not groups or len(groups) > 256:
            raise ValueError("Expected 1..256 packed chunks")
        owned, refs = [], {}
        with self.lock:
            for previous in releases:
                self._finish(previous)
                self.counts["release_calls"] += 1
            if self.closing:
                raise RuntimeError("SHM reader is closing")
            self._reap()
            if token in self.leases or token in self.finished:
                raise ValueError("Duplicate or cancelled lease")
            if len(self.leases) >= 4096:
                raise RuntimeError("Consumer lease capacity exhausted")
            missing = [g for g in groups if g not in self.entries]
            needed = len(missing) - len(self.free)
            if needed > 0:
                requested, victims = set(groups), []
                for group, entry in self.entries.items():
                    if group not in requested and not entry.refs and entry.ready.done():
                        victims.append(group)
                        if len(victims) >= needed:
                            break
                for group in victims:
                    self._drop(group, "capacity_evictions")
            if len(self.free) < len(missing):
                raise RuntimeError("SHM slots busy; no live lease may be evicted")
            touched = time.monotonic()
            for group in groups:
                entry = self.entries.get(group)
                if entry is None:
                    slot = self.free.popleft()
                    self.generations[slot] += 1
                    entry = Entry(slot, self.generations[slot])
                    self.entries[group] = entry
                    owned.append((group, entry))
                else:
                    self.counts[
                        "reuse_hits" if entry.ready.done() else "singleflight_joins"
                    ] += 1
                entry.refs += 1
                entry.touched = touched
                self.entries.move_to_end(group)
                refs[group] = entry
            self.leases[token] = dict(pid=pid, cookie=cookie, entries=refs)
            if owned:
                self.active += 1
                self.executor.submit(self._load, owned)
        # Native I/O and readiness waits retain the original lifetime contract.
        ready = {group: entry.ready.result(timeout=25) for group, entry in refs.items()}
        with self.lock:
            if token not in self.leases:
                raise RuntimeError("Lease was released before completion")
            snapshot = {
                g: (e.slot * PACK_SIZE, e.generation, ready[g]) for g, e in refs.items()
            }
            self.counts["get_batches"] += 1
            self.counts["get_objects"] += len(rows)
            self.counts["logical_get_bytes"] += sum(length for _, _, length in plan)
        # Entries are immutable while this lease is held. Cancellation invalidates
        # the response just as it does after the original in-lock return.
        result = []
        for group, component_offset, length in plan:
            base, generation, ok = snapshot[group]
            result.append([base + component_offset, length, generation] if ok else None)
        return dict(
            version=1,
            pool=self.name,
            pool_bytes=self.capacity,
            lease=token,
            items=result,
            ready=True,
        )

    def release_many(self, tokens: list[str]) -> None:
        """Release completed window leases under one pool lock."""
        with self.lock:
            for token in tokens:
                self._finish(token)
                self.counts["release_calls"] += 1

    def release(self, token: str) -> Any:
        """Release a completed lease, or record cancellation before admission."""
        with self.lock:
            self._finish(token)  # Also cancels a not-yet-admitted request.
            self.counts["release_calls"] += 1
        return {"released": True}

    def clear(self) -> Any:
        """Drop cached packs only after all I/O and consumer leases have drained."""
        with self.lock:
            self._reap()
            if self.active or self.leases:
                raise RuntimeError("Cannot clear live shared reads")
            for group in list(self.entries):
                self._drop(group)
        return self.status()

    def invalidate(self, groups: list[str]) -> Any:
        """Drop unleased packs before a write; reject still-owned entries."""
        with self.lock:
            for group in groups:
                e = self.entries.get(group)
                if e is not None:
                    if e.refs or not e.ready.done():
                        raise RuntimeError("Cannot overwrite a leased pack")
                    self._drop(group)

    def status(self) -> dict[str, Any]:
        """Report pool ownership and readiness without checking process liveness."""
        with self.lock:
            self._reap()
            return dict(
                pool_name=self.name,
                pool_bytes=self.capacity,
                pool_slots=len(self.generations),
                used_slots=len(self.entries),
                quarantined_leases=sum(
                    bool(x.get("quarantined")) for x in self.leases.values()
                ),
                leases=len(self.leases),
                leased_slots=sum(e.refs > 0 for e in self.entries.values()),
                loading_batches=self.active,
                **self.counts,
            )

    def close(self) -> None:
        """Drain owned work before releasing its resources."""
        with self.lock:
            self.closing = True
        self.executor.shutdown(wait=True)
        # Unlink does not invalidate mappings in existing clients; no new owner
        # ever reuses this randomly named pool after a server restart.
        self.mapping.close()
        shm_munmap(self.address)
        shm_unlink(self.name)
