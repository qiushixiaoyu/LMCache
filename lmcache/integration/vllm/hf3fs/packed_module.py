# SPDX-License-Identifier: Apache-2.0
"""One public LMCache HF3FS L2 object per complete canonical token chunk.

The existing bounded direct module supplies admission, timeout ownership and
shutdown draining. Packed files use a separate directory and namespace. No
payload or FD cache exists here; bridge singleflight coalesces GPU consumers.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any
import hashlib
import threading
import time

# Third Party
import msgpack
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj
from lmcache.v1.multiprocess.ext_server_module import server_module_handler

# Local
from .canonical_direct_module import CanonicalObjectModule
from .pack_format import (
    LAYOUT_HASH,
    PACK_SIZE,
    checked_groups,
    group_key,
    pack,
    split_key,
    unpack,
)


class PackedCanonicalModule(CanonicalObjectModule):
    """Restore complete packs once and return requested canonical components."""

    def __init__(self, ctx: Any, config: dict[str, Any], adapter: Any) -> None:
        super().__init__(ctx, config, adapter)
        self.store_lock = threading.Lock()

    @server_module_handler("canonical.put")
    def put(self, payload: bytes) -> bytes:
        """Acknowledge only complete packs whose native store has finished."""
        return self._call("put", payload)

    def report_status(self) -> dict[str, Any]:
        """Return current resource counts and storage completion metrics."""
        result = super().report_status()
        result["canonical_objects"]["read_path"] = "packed_l2_v1"
        result["canonical_objects"]["pack_bytes"] = PACK_SIZE
        return result

    def keys(self, groups: list[str]) -> Any:
        """Encode complete pack identities as LMCache object keys."""
        return [
            ObjectKey(
                chunk_hash=hashlib.sha256(group.encode()).digest(),
                model_name="dsv41-pack-v1-" + LAYOUT_HASH.hex()[:40],
                kv_rank=ObjectKey.ComputeKVRank(
                    world_size=1, global_rank=0, local_world_size=1, local_rank=0
                ),
            )
            for group in groups
        ]

    def buffers(self, count: int) -> Any:
        """Allocate CPU pack storage and views kept alive through native completion."""
        buffer = torch.empty(count * PACK_SIZE, dtype=torch.uint8, device="cpu")
        objects = []
        for i in range(count):
            view = buffer[i * PACK_SIZE : (i + 1) * PACK_SIZE]
            meta = MemoryObjMetadata(
                shape=torch.Size([PACK_SIZE]),
                dtype=torch.uint8,
                address=view.data_ptr(),
                phy_size=PACK_SIZE,
                ref_count=1,
            )
            objects.append(TensorMemoryObj(view, meta, parent_allocator=None))
        return buffer, objects

    def _exists(self, rows: list[list[Any]]) -> Any:
        groups = list(checked_groups("exists", rows))
        keys = self.keys(groups)
        task = self.adapter.submit_lookup_and_lock_task(keys, {})
        result = None
        while result is None:
            result = self.adapter.query_lookup_and_lock_result(task)
            if result is None:
                time.sleep(0.0005)
        indices = result.get_indices_list()
        hits = {groups[i] for i in indices}
        self.adapter.submit_unlock([keys[i] for i in indices])
        with self.lock:
            self.counts["metadata_objects"] += len(groups)
            self.counts["logical_metadata_objects"] += len(rows)
        return [int(group_key(row[0]) in hits) for row in rows]

    def _get(self, rows: list[list[Any]]) -> Any:
        groups = list(checked_groups("get", rows))
        # Bound physical allocation too, even for sparse component requests.
        if len(groups) > 32:
            raise ValueError("At most 32 packed chunks per native batch")
        buffer, objects = self.buffers(len(groups))
        with self.lock:
            self.buffer_bytes += buffer.numel()
        try:
            start = time.monotonic()
            task = self.adapter.submit_load_task(self.keys(groups), objects)
            result = None
            while result is None:
                result = self.adapter.query_load_result(task)
                if result is None:
                    time.sleep(0.0005)
            elapsed = time.monotonic() - start
            values = {}
            for i in result.get_indices_list():
                try:
                    values[groups[i]] = unpack(
                        groups[i],
                        objects[i].raw_data.numpy().tobytes(),
                    )
                except ValueError:
                    with self.lock:
                        self.counts["corrupt_packs"] += 1
            with self.lock:
                self.counts["native_get_batches"] += 1
                self.counts["native_get_objects"] += len(groups)
                self.counts["loaded_objects"] += len(values)
                self.counts["loaded_bytes"] += len(values) * PACK_SIZE
                self.counts["native_load_seconds"] += elapsed
                self.counts["missing_objects"] += len(groups) - len(values)
            return [
                values.get(group_key(row[0]), {}).get(split_key(row[0])[1])
                for row in rows
            ]
        finally:
            with self.lock:
                self.buffer_bytes -= buffer.numel()

    def _put(self, rows: list[list[Any]]) -> Any:
        groups = checked_groups("put", rows)
        if len(groups) > 32:
            raise ValueError("At most 32 packed chunks per native batch")
        # Validate the entire batch before submitting any writes.
        data = [
            pack(group, list(components.values()))
            for group, components in groups.items()
        ]
        buffer, objects = self.buffers(len(groups))
        for obj, value in zip(objects, data, strict=True):
            obj.raw_data.copy_(torch.frombuffer(bytearray(value), dtype=torch.uint8))
        with self.lock:
            self.buffer_bytes += buffer.numel()
        try:
            # Store result polling drains a shared completion queue.
            with self.store_lock:
                task = self.adapter.submit_store_task(self.keys(groups), objects)
                result = None
                while result is None:
                    completed = self.adapter.pop_completed_store_tasks()
                    if completed and set(completed) != {task}:
                        raise RuntimeError("Unexpected store completion")
                    result = completed.get(task)
                    if result is None:
                        time.sleep(0.0005)
                if not result.is_successful():
                    raise RuntimeError("Packed native store failed")
            with self.lock:
                self.counts["stored_objects"] += len(groups)
                self.counts["stored_bytes"] += len(groups) * PACK_SIZE
            return [0] * len(rows)
        finally:
            with self.lock:
                self.buffer_bytes -= buffer.numel()

    def _run(self, operation: str, rows: list[list[Any]]) -> Any:
        if operation != "put":
            return super()._run(operation, rows)

        start = time.monotonic()
        with self.lock:
            self.active += 1
        try:
            result = self._put(rows)
            with self.lock:
                self.counts["put_batches"] += 1
                self.counts["put_objects"] += len(rows)
                self.counts["put_seconds"] += time.monotonic() - start
            return msgpack.packb(result, use_bin_type=True)
        finally:
            with self.lock:
                self.active -= 1
