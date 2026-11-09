# SPDX-License-Identifier: Apache-2.0
"""One descriptor GET per window; leases survive every source-group copy."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import Future
from typing import Any, cast
import time

# First Party
from lmcache.v1.platform.devices.cuda.hf3fs_copy import register


class Windows:
    def __init__(self, clients: Any) -> None:
        if not clients or any(c.channel != "unix" for c in clients):
            raise ValueError("Merged windows require Unix descriptor clients")
        self.clients = clients
        self.held: list[tuple[Any, str, list[memoryview | None]]] = []
        self.name = None

    def get(self, index: int, rows: list[list[Any]]) -> Any:
        """Obtain pack views; the consumer must release them after its final copy
        completes.
        """
        client = self.clients[index % len(self.clients)]
        data = client.get(rows)
        # Ownership moves from the reusable connection to this load session.
        self.held.append((client, client.lease, data))
        client.lease = None
        if self.name is None:
            self.name = client.name
        if client.name != self.name:
            raise RuntimeError("Shared pool changed during a layered load")
        if any(
            v is None or len(v) != row[1] for v, row in zip(data, rows, strict=True)
        ):
            raise RuntimeError("Incomplete merged-window cache load")
        return data

    def close(self) -> None:
        # Caller must have proved that all native/CUDA work using these views
        # has completed. Missing completion deliberately retains this session.
        """Release window views after the caller has proved all copies complete."""
        for client, token, data in self.held:
            for value in data:
                if value is not None:
                    value.release()
        for client in self.clients:
            tokens = [token for owner, token, _ in self.held if owner is client]
            for offset in range(0, len(tokens), 128):
                client.call("release_shm", {"leases": tokens[offset : offset + 128]})
            client.release(force=True)
        self.held.clear()


def load(store: Any, batches: Any, sources: Any, ready: Any) -> Any:
    """Preserve source-major gates and the existing two-window I/O pipeline."""

    with store.io_lock:
        if store.poisoned:
            raise RuntimeError("Previous CUDA completion failed")
        if getattr(store, "merged_session", None) is not None:
            raise RuntimeError("Previous merged-window session has not retired")
        session = Windows(store.prefetch_clients)
        # Keep the mapping, tokens and views reachable on failed CUDA completion.
        store.merged_session = session
        future = None
        started = time.perf_counter()
        total_keys = sum(len(b[0]) for b in batches)
        total_bytes = sum(sum(map(sum, b[2])) for b in batches)
        data_by_window = {}

        def fetch(index: int) -> Any:
            keys, _, sizes = batches[index]
            return session.get(
                index, [[k, sum(s)] for k, s in zip(keys, sizes, strict=True)]
            )

        try:
            if batches:
                future = store.prefetch.submit(fetch, 0)
            for source_index, source in enumerate(sources):
                for index, (keys, addresses, sizes) in enumerate(batches):
                    if source_index == 0:
                        waiting = time.perf_counter()
                        data = cast(Future[Any], future).result()
                        future = None
                        data_by_window[index] = data
                        client = store.prefetch_clients[index % 2]
                        if store.trace:
                            store.counts["prefetch_wait_seconds"] += (
                                time.perf_counter() - waiting
                            )
                            store.counts.update(client.counts)
                        if index + 1 < len(batches):
                            future = store.prefetch.submit(fetch, index + 1)
                        if not client.registered:
                            client.registered = register(client.mapping)
                    else:
                        data = data_by_window[index]
                    selected = [
                        i for i, k in enumerate(keys) if f"@layer_{source}/" in k
                    ]
                    if selected:
                        store.copy_direct(
                            [data[i] for i in selected],
                            [addresses[i] for i in selected],
                            [sizes[i] for i in selected],
                        )
                        if store.trace:
                            store.counts.update(
                                batches=1,
                                objects=len(selected),
                                regions=sum(len(sizes[i]) for i in selected),
                                bytes=sum(sum(sizes[i]) for i in selected),
                            )
                ready(source)
            store.counts.update(
                merged_loads=1,
                merged_descriptor_gets=len(batches),
                merged_components=total_keys,
            )
            return total_keys, total_bytes
        finally:
            try:
                # Native reads retain their lease even when another window fails.
                if future is not None:
                    future.result()
            finally:
                if not store.poisoned:
                    session.close()
                    store.merged_session = None
                if store.trace:
                    store.counts["total_get_seconds"] += time.perf_counter() - started
