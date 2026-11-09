# SPDX-License-Identifier: Apache-2.0
"""Public lease/lifetime regressions using the installed CPU SHM allocator."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import os
import subprocess
import threading
import time
import unittest
import uuid

# First Party
from lmcache.integration.vllm.hf3fs.pack_format import COMPONENTS, descriptor
from lmcache.integration.vllm.hf3fs.shm_pool import PackPool, process_cookie


def rows(i, components=COMPONENTS):
    group = "dsv41_canonical_v1@" + "a" * 64 + "@" + f"{i:064x}"
    return [descriptor(group, c) for c in components]


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.pool = PackPool(lambda entries: {g for g, _ in entries}, slots=2)
        self.pid = os.getpid()
        self.cookie = process_cookie(self.pid)

    def tearDown(self):
        self.pool.close()

    def get(self, i, token=None, releases=()):
        return self.pool.acquire(
            rows(i), token or uuid.uuid4().hex, self.pid, self.cookie, releases=releases
        )

    def until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)

    def test_long_history_window_leases(self):
        held = [self.get(1) for _ in range(1024)]
        self.assertEqual(self.pool.status()["leases"], 1024)
        self.assertEqual(self.pool.status()["leased_slots"], 1)
        for lease in held:
            self.pool.release(lease["lease"])
        self.assertEqual(self.pool.status()["leased_slots"], 0)

    def test_release_and_acquire_full_pool(self):
        first, held = self.get(1), self.get(2)
        with self.assertRaises(RuntimeError):
            self.get(3)
        third = self.get(3, releases=[first["lease"]])
        self.assertEqual(third["items"][0][2], 2)
        self.assertEqual(self.pool.status()["leases"], 2)
        self.pool.release(first["lease"])
        self.pool.release(third["lease"])
        self.pool.release(held["lease"])
        self.assertEqual(self.pool.status()["leased_slots"], 0)

    def test_shared_pack_and_component_offsets(self):
        a, b = self.get(1), self.get(1)
        self.assertEqual(a["items"], b["items"])
        self.assertEqual(self.pool.status()["used_slots"], 1)
        for left, right in zip(a["items"], a["items"][1:], strict=False):
            self.assertEqual(left[0] + left[1], right[0])
        self.pool.release(a["lease"])
        self.assertEqual(self.pool.status()["leased_slots"], 1)
        self.pool.release(b["lease"])
        self.pool.clear()

    def test_late_cancel_is_idempotent(self):
        token = uuid.uuid4().hex
        self.pool.release(token)
        self.pool.release(token)
        with self.assertRaises(ValueError):
            self.get(1, token)

    def test_cancel_during_io_retains_slot(self):
        gate = threading.Event()
        self.pool.close()
        self.pool = PackPool(
            lambda entries: gate.wait(5) and {g for g, _ in entries}, slots=1
        )
        token = uuid.uuid4().hex
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(self.get, 1, token)
            try:
                self.until(lambda: self.pool.status()["loading_batches"] == 1)
                self.pool.release(token)
                with self.assertRaises(RuntimeError):
                    self.get(2)
                with self.assertRaises(RuntimeError):
                    self.pool.clear()
            finally:
                gate.set()
            with self.assertRaises(RuntimeError):
                future.result()
        self.until(lambda: self.pool.status()["loading_batches"] == 0)
        result = self.get(2)
        self.pool.release(result["lease"])
        self.pool.clear()

    def test_concurrent_singleflight(self):
        gate, calls = threading.Event(), []

        def loader(entries):
            calls.append(entries)
            assert gate.wait(5)
            return {g for g, _ in entries}

        self.pool.loader = loader
        with ThreadPoolExecutor(2) as executor:
            a, b = executor.submit(self.get, 1), executor.submit(self.get, 1)
            try:
                self.until(lambda: self.pool.status()["leases"] == 2)
            finally:
                gate.set()
            first, second = a.result(), b.result()
        self.assertEqual(len(calls), 1)
        self.pool.release(first["lease"])
        self.pool.release(second["lease"])

    def test_failed_read_can_retry(self):
        self.pool.loader = lambda entries: set()
        first = self.get(1)
        self.assertTrue(all(x is None for x in first["items"]))
        self.pool.release(first["lease"])
        self.pool.loader = lambda entries: {g for g, _ in entries}
        second = self.get(1)
        self.assertTrue(all(x is not None for x in second["items"]))
        self.pool.release(second["lease"])

    def test_loader_exception_can_retry(self):
        def fail(entries):
            raise OSError("synthetic IO failure")

        self.pool.loader = fail
        first = self.get(1)
        self.assertTrue(all(x is None for x in first["items"]))
        self.pool.release(first["lease"])
        self.pool.loader = lambda entries: {g for g, _ in entries}
        second = self.get(1)
        self.assertTrue(all(x is not None for x in second["items"]))
        self.pool.release(second["lease"])

    def test_zero_ttl_does_not_expire_live_lease(self):
        self.pool.ttl = 0
        first = self.get(1)
        self.assertEqual(self.pool.status()["leased_slots"], 1)
        self.pool.release(first["lease"])
        self.assertEqual(self.pool.status()["used_slots"], 0)

    def test_acquire_does_not_check_process_start_time(self):
        # Existing clients may still send a start-time cookie. It is metadata,
        # not a reason to perform a /proc read on every descriptor GET.
        with patch(
            "lmcache.integration.vllm.hf3fs.shm_pool.process_cookie",
            side_effect=AssertionError("unexpected process polling"),
        ):
            first = self.pool.acquire(
                rows(1), uuid.uuid4().hex, self.pid, "opaque-cookie"
            )
            second = self.pool.acquire(
                rows(1), uuid.uuid4().hex, self.pid, "different-cookie"
            )
            self.assertEqual(self.pool.status()["leases"], 2)
            self.assertEqual(first["items"], second["items"])
            self.pool.release(first["lease"])
            self.assertEqual(self.pool.status()["leased_slots"], 1)
            self.pool.release(second["lease"])
            self.assertEqual(self.pool.status()["leased_slots"], 0)

    def test_dead_consumer_remains_pinned_without_polling(self):
        child = subprocess.Popen(["sleep", "30"])
        try:
            held = self.pool.acquire(
                rows(1), uuid.uuid4().hex, child.pid, process_cookie(child.pid)
            )
        finally:
            child.terminate()
            child.wait()
        # Status must not probe any consumer. The held references alone protect it.
        with patch(
            "lmcache.integration.vllm.hf3fs.shm_pool.process_cookie",
            side_effect=AssertionError("unexpected process polling"),
        ):
            state = self.pool.status()
            self.assertEqual(state["leases"], 1)
            self.assertEqual(state["quarantined_leases"], 0)
            with self.assertRaises(RuntimeError):
                self.pool.clear()
        # Synthetic child submitted no DMA; explicit cleanup is valid here.
        self.pool.release(held["lease"])
        self.pool.clear()


if __name__ == "__main__":
    unittest.main(verbosity=2)
