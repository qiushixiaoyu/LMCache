# SPDX-License-Identifier: Apache-2.0
"""Canonical SHM retrieve extension: KV payload stays inside a node-local pool."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import os
import time
import uuid

# Third Party
import msgpack
import torch

# First Party
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.multiprocess.ext_server_module import server_module_handler

# Local
from .pack_format import PACK_SIZE, checked_groups
from .pack_validation import valid_pack
from .packed_module import PackedCanonicalModule
from .shm_socket import Server


class SharedCanonicalModule(PackedCanonicalModule):
    def __init__(
        self, ctx: Any, config: dict[str, Any], adapter: Any, shared: Any, link: Any
    ) -> None:
        super().__init__(ctx, config, adapter)
        self.verify_pack_hashes = config.get("verify_pack_hashes", False)
        if type(self.verify_pack_hashes) is not bool:
            raise ValueError("verify_pack_hashes must be boolean")
        self.shared, self.iov_link = shared, link
        self.slot_objects = []
        for slot in range(shared.capacity // PACK_SIZE):
            view = torch.frombuffer(
                shared.mapping,
                dtype=torch.uint8,
                count=PACK_SIZE,
                offset=slot * PACK_SIZE,
            )
            meta = MemoryObjMetadata(
                shape=torch.Size([PACK_SIZE]),
                dtype=torch.uint8,
                address=view.data_ptr(),
                phy_size=PACK_SIZE,
                ref_count=1,
            )
            self.slot_objects.append(TensorMemoryObj(view, meta, parent_allocator=None))
        self.shared.loader = self._load_shared

        self.socket = Server(
            self, config.get("socket_path", "/tmp/lmcache-hf3fs-read.sock")
        )

    def _load_shared(self, entries: Any) -> Any:
        started = time.perf_counter()
        verify_hashes = self.verify_pack_hashes
        objects = [self.slot_objects[slot] for _, slot in entries]
        with self.lock:
            self.buffer_bytes += len(entries) * PACK_SIZE
        try:
            prepared = time.perf_counter()
            task = self.adapter.submit_load_task(
                self.keys([g for g, _ in entries]), objects
            )
            result = None
            while result is None:
                result = self.adapter.query_load_result(task)
                if result is None:
                    time.sleep(0.0005)
            loaded = time.perf_counter()
            valid, corrupt = set(), 0
            for i in result.get_indices_list():
                group, slot = entries[i]
                data = memoryview(self.shared.mapping)[
                    slot * PACK_SIZE : (slot + 1) * PACK_SIZE
                ]
                try:
                    if valid_pack(group, data, verify_hashes):
                        valid.add(group)
                    else:
                        corrupt += 1
                finally:
                    data.release()
            checked = time.perf_counter()
            with self.lock:
                self.counts.update(
                    native_get_batches=1,
                    native_get_objects=len(entries),
                    loaded_objects=len(valid),
                    loaded_bytes=len(valid) * PACK_SIZE,
                    missing_objects=len(entries) - len(valid),
                    corrupt_packs=corrupt,
                    native_load_seconds=loaded - prepared,
                    pack_validation_seconds=checked - loaded,
                    checksum_seconds=(checked - loaded) if verify_hashes else 0,
                    hash_checked_packs=len(result.get_indices_list())
                    if verify_hashes
                    else 0,
                    hash_skipped_packs=0
                    if verify_hashes
                    else len(result.get_indices_list()),
                    buffer_prepare_seconds=prepared - started,
                )
            return valid
        finally:
            with self.lock:
                self.buffer_bytes -= len(entries) * PACK_SIZE

    def _get(self, rows: list[list[Any]]) -> Any:
        """Preserve the legacy byte API using the same registered read arena."""
        token = uuid.uuid4().hex
        try:
            response = self.shared.acquire(rows, token, os.getpid(), None)
            return [
                None
                if item is None
                else bytes(self.shared.mapping[item[0] : item[0] + item[1]])
                for item in response["items"]
            ]
        finally:
            self.shared.release(token)

    @server_module_handler("canonical.get_shm")
    def get_shm(self, payload: bytes) -> bytes:
        if len(payload) > 1024**2:
            raise ValueError("Oversized descriptor request")
        request = msgpack.unpackb(payload, raw=False)
        for token in request.get("releases", []):
            self.shared.release(token)
        rows = request["rows"]
        if not isinstance(rows, list) or not 0 < len(rows) <= 4096:
            raise ValueError("Invalid bounded descriptor batch")
        for offset in range(0, len(rows), 256):
            checked_groups("get", rows[offset : offset + 256])
        if sum(r[1] for r in rows) > 64 * 1024**2:
            raise ValueError("Logical batch byte budget exceeded")
        result = self.shared.acquire(
            rows, request["lease"], request["pid"], request["cookie"]
        )
        encoded = msgpack.packb(result, use_bin_type=True)
        with self.lock:
            self.counts["shm_rpc_metadata_bytes"] += len(encoded)
            self.counts["shm_rpc_kv_payload_bytes"] += 0
        return encoded

    @server_module_handler("canonical.release_shm")
    def release_shm(self, payload: bytes) -> bytes:
        if len(payload) > 256:
            raise ValueError("Oversized release")
        request = msgpack.unpackb(payload, raw=False)
        tokens = request.get("leases", [request.get("lease")])
        if len(tokens) > 4 or any(
            not isinstance(t, str) or len(t) != 32 for t in tokens
        ):
            raise ValueError("Invalid leases")
        for token in tokens:
            self.shared.release(token)
        return msgpack.packb({"released": True}, use_bin_type=True)

    @server_module_handler("canonical.shm_control")
    def shm_control(self, payload: bytes) -> bytes:
        request = msgpack.unpackb(payload, raw=False)
        action = request["action"]
        if action == "clear":
            result = self.shared.clear()
        elif action == "status":
            result = self.shared.status()
        elif action == "set_pack_hash_checks":
            enabled = request["enabled"]
            if type(enabled) is not bool:
                raise ValueError("enabled must be boolean")
            # The same lock guards admission. No in-flight read or retained
            # result may cross a policy change.
            with self.shared.lock:
                self.shared.clear()
                self.verify_pack_hashes = enabled
                result = self.shared.status()
                result["verify_pack_hashes"] = enabled
        else:
            raise ValueError(action)
        return msgpack.packb(result, use_bin_type=True)

    def _put(self, rows: list[list[Any]]) -> Any:
        # Canonical packs are immutable. Reject overwrites while leased.
        self.shared.invalidate(checked_groups("put", rows))
        return super()._put(rows)

    def report_status(self) -> dict[str, Any]:
        """Return current resource counts and storage completion metrics."""
        result = super().report_status()
        if hasattr(self, "shared"):
            result["canonical_objects"]["shared"] = self.shared.status()
            result["canonical_objects"]["verify_pack_hashes"] = self.verify_pack_hashes
            result["canonical_objects"]["pack_reader_revision"] = (
                "pack-hash-optional-v1"
            )
        return result

    def close(self) -> None:
        """Drain owned work before releasing buffers, handles and completion resources."""
        self.socket.close()
        self.shared.executor.shutdown(wait=True)
        super().close()
        self.slot_objects.clear()
        self.iov_link.unlink()
        self.shared.close()
