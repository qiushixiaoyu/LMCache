# SPDX-License-Identifier: Apache-2.0
"""Prefix proof and merged-window ownership tests without GPU operations."""

# Standard
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.hf3fs.prefix_progress import PositiveObjects, PrefixIndex
from lmcache.integration.vllm.hf3fs.window_reader import load


def test_prefix_requires_contiguous_ack() -> None:
    hashes = [hashlib.sha256(str(i).encode()).digest() for i in range(16)]
    index = PrefixIndex(128, 64)
    assert not index.confirm(hashes, 128, 256)
    assert index.longest(hashes, 1024) == 0
    assert index.confirm(hashes, 0, 256)
    assert index.confirm(hashes, 256, 512)
    assert index.longest(hashes, 511) == 384
    assert index.longest([b"other"] * 16, 1024) == 0


def test_positive_cache_does_not_remember_misses() -> None:
    index = PositiveObjects(2)
    calls = []

    def query(keys: list[str]) -> list[int]:
        calls.append(keys)
        return [int(key != "missing") for key in keys]

    assert index.exists(["a", "a", "missing"], query)[0] == [1, 1, 0]
    assert index.exists(["a", "missing"], query) == ([1, 0], 1)
    assert calls == [["a", "missing"], ["missing"]]
    with pytest.raises(RuntimeError):
        index.exists(["failed"], lambda keys: [-1])


class DescriptorClient:
    channel = "unix"
    name = "test-pool"
    registered = 1
    mapping = None
    lease = None
    counts = {}

    def __init__(self) -> None:
        self.gets = 0
        self.releases = []

    def get(self, rows: list[list]) -> list[memoryview]:
        self.gets += 1
        self.lease = str(self.gets)
        return [memoryview(bytearray([i + 1] * row[1])) for i, row in enumerate(rows)]

    def call(self, operation: str, value: dict) -> None:
        assert operation == "release_shm"
        self.releases.extend(value["leases"])

    def release(self, force: bool = False) -> None:
        assert self.lease is None


def run_windows(fail: bool) -> tuple:
    clients = [DescriptorClient(), DescriptorClient()]
    sources = [2, 8, 14, 20]
    keys = [
        f"ns@layer_{source}/{kind}@prefix"
        for source in sources
        for kind in ("main_kv", "indexer_k")
    ]
    batches = [(keys, [[i] for i in range(8)], [[4]] * 8)] * 2
    ready = []
    copies = []
    with ThreadPoolExecutor(1) as executor:
        store = SimpleNamespace(
            io_lock=threading.RLock(),
            poisoned=False,
            merged_session=None,
            prefetch_clients=clients,
            prefetch=executor,
            trace=False,
            counts=Counter(),
        )

        def copy(data: list[memoryview], addresses: list, sizes: list) -> None:
            if fail:
                store.poisoned = True
                raise RuntimeError("CUDA completion unproven")
            copies.append([bytes(v) for v in data])

        store.copy_direct = copy
        with patch(
            "lmcache.integration.vllm.hf3fs.window_reader.register",
            side_effect=AssertionError("GPU call"),
        ):
            if fail:
                with pytest.raises(RuntimeError, match="unproven"):
                    load(store, batches, sources, ready.append)
            else:
                assert load(store, batches, sources, ready.append) == (16, 64)
    return clients, store, ready, copies


def test_one_descriptor_get_per_window_and_layer_release() -> None:
    clients, store, ready, copies = run_windows(False)
    assert sum(c.gets for c in clients) == 2
    assert sum(len(c.releases) for c in clients) == 2
    assert ready == [2, 8, 14, 20]
    assert len(copies) == 8
    assert store.merged_session is None


def test_unproven_gpu_completion_retains_all_window_leases() -> None:
    clients, store, ready, copies = run_windows(True)
    assert not ready and not copies
    assert all(not c.releases for c in clients)
    assert store.merged_session is not None
    assert all(len(v) == 4 for _, _, data in store.merged_session.held for v in data)
    # This synthetic copy submitted no DMA; explicitly close after that fact is known.
    store.merged_session.close()
