# SPDX-License-Identifier: Apache-2.0
"""Bounded canonical reads through LMCache's public HF3FS L2 adapter.

Admission and timeouts preserve native buffer ownership until completion.
Subclasses supply canonical pack lookup, read and write operations.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any
import threading
import time

# Third Party
import msgpack

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import DiscoverableModule
from lmcache.v1.multiprocess.ext_server_module import server_module_handler

# Local
from .canonical_protocol import MAX_BODY, validate

logger = init_logger(__name__)


class CanonicalObjectModule(DiscoverableModule):
    """Own a read adapter and at most ``concurrency`` outstanding batch buffers."""

    module_name = "canonical_objects"

    def __init__(
        self,
        ctx: MPCacheServerContext,
        config: dict[str, Any],
        adapter: L2AdapterInterface,
    ) -> None:
        self._ctx = ctx
        self.adapter = adapter
        self.timeout = float(config.get("timeout", 20))
        concurrency = int(config.get("concurrency", 4))
        if concurrency <= 0 or self.timeout <= 0:
            raise ValueError("Concurrency and timeout must be positive")
        self.slots = threading.BoundedSemaphore(concurrency)
        self.pool = ThreadPoolExecutor(concurrency, thread_name_prefix="canonical-read")
        self.lock = threading.Lock()
        self.counts: Any = Counter()
        self.active = 0
        self.buffer_bytes = 0
        self.closing = False
        self.orphans: set[Future[bytes]] = set()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the MP context used for canonical object identity."""
        return self._ctx

    def report_status(self) -> dict[str, Any]:
        """Return reader health, bounded resource use, and actual I/O counts."""
        adapter_status = self.adapter.report_status()
        with self.lock:
            return {
                "canonical_objects": {
                    "is_healthy": not self.closing and adapter_status["is_healthy"],
                    "read_path": "direct_l2",
                    "active": self.active,
                    "pending_cleanup": len(self.orphans),
                    "buffer_bytes": self.buffer_bytes,
                    "reader": adapter_status,
                    **self.counts,
                }
            }

    def close(self) -> None:
        """Drain native work before releasing its buffers and adapter.

        A caller timeout does not cancel I/O. Shutdown waits for outstanding
        reads, including reads whose RPC caller already received a timeout.
        """
        with self.lock:
            self.closing = True
        self.pool.shutdown(wait=True)
        self.adapter.close()

    @server_module_handler("canonical.exists")
    def exists(self, payload: bytes) -> bytes:
        """Return fresh presence bits for validated canonical descriptors."""
        return self._call("exists", payload)

    @server_module_handler("canonical.get")
    def get(self, payload: bytes) -> bytes:
        """Return bytes or None per descriptor, preserving duplicates/order.

        Missing, deleted, or short-read objects return None. Successful bytes
        are a file snapshot held by native open through completion. RPC timeout
        retains the worker slot and its buffers until the native task finishes.
        """
        return self._call("get", payload)

    def _completed(self, future: Future[bytes]) -> None:
        with self.lock:
            orphan = future in self.orphans
            self.orphans.discard(future)
        if orphan and future.exception() is not None:
            logger.error(
                "Timed-out canonical task subsequently failed: %s", future.exception()
            )
        self.slots.release()

    def _run(self, operation: str, rows: list[list[Any]]) -> bytes:
        started = time.monotonic()
        with self.lock:
            self.active += 1
        try:
            result = self._exists(rows) if operation == "exists" else self._get(rows)
            with self.lock:
                self.counts[operation + "_batches"] += 1
                self.counts[operation + "_objects"] += len(rows)
                self.counts[operation + "_seconds"] += time.monotonic() - started
            return msgpack.packb(result, use_bin_type=True)
        finally:
            with self.lock:
                self.active -= 1

    def _call(self, operation: str, payload: bytes) -> bytes:
        if len(payload) > MAX_BODY:
            raise ValueError("Oversized extension payload")
        rows = msgpack.unpackb(payload, raw=False)
        validate(operation, rows)
        if not self.slots.acquire(timeout=self.timeout):
            raise LMCacheTimeoutError("Canonical reader capacity exhausted")
        with self.lock:
            if self.closing:
                self.slots.release()
                raise RuntimeError("Canonical reader is closing")
            # Serialize submission with shutdown so every accepted batch is
            # included in the executor drain.
            future = self.pool.submit(self._run, operation, rows)
        future.add_done_callback(self._completed)
        try:
            return future.result(timeout=self.timeout)
        except TimeoutError:
            with self.lock:
                if not future.done():
                    self.orphans.add(future)
                self.counts["rpc_timeouts"] += 1
            raise
