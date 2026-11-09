# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import ctypes
import os
import uuid

# Third Party
import msgpack

# First Party
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.multiprocess.ext_server_module import server_module_handler

# Local
from .lock_shm_module import LockSharedModule
from .packed_module import PackedCanonicalModule
from .shm_module import SharedCanonicalModule
from .shm_pool import PackPool
from .slot_client import SlotClient
from .slot_index import PACK_BYTES, SLOT_BYTES


def adapt(client: Any) -> Any:
    """Expose the slot completion API through the native LMCache L2 adapter."""
    return NativeConnectorL2Adapter(
        client,
        MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE),
        type_name="SlotHf3fs",
    )


class Status:
    def report_status(self) -> dict[str, Any]:
        """Return current resource counts and storage completion metrics."""
        result = (
            SharedCanonicalModule.report_status(self)
            if hasattr(self, "shared")
            else PackedCanonicalModule.report_status(self)
        )
        result["canonical_objects"]["native_metadata"] = self.client.metadata_stats()
        result["canonical_objects"]["slot_policy"] = dict(
            version=1,
            immutable_no_eviction=True,
            pack_bytes=PACK_BYTES,
            slot_bytes=SLOT_BYTES,
            index="foundationdb",
            legacy_read=True,
            legacy_writes=self.client.legacy_writes,
            index_cache_entries=self.client.index.cache_entries,
            config=self.client.index.config,
        )
        return result

    @server_module_handler("canonical.fd_control")
    def fd_control(self, payload: bytes) -> bytes:
        """Inspect or clear registered handles without invalidating in-flight leases."""
        action = msgpack.unpackb(payload, raw=False)["action"]
        if action == "clear":
            self.client.clear_fd_cache()
        elif action != "status":
            raise ValueError(action)
        return msgpack.packb(self.client.metadata_stats(), use_bin_type=True)


class SlotSharedModule(Status, LockSharedModule):
    @classmethod
    def create(cls, ctx: Any) -> Any:
        """Construct the configured adapter and its owned resources."""
        config = dict(ctx.config_for(cls))
        cfg = config["reader_adapter"]
        shared = PackPool(None, config.get("shm_slots", 131072))
        identity = uuid.uuid4().bytes
        link = (
            Path(cfg["mount_point"]) / "3fs-virt/iovs" / str(uuid.UUID(bytes=identity))
        )
        client = None
        try:
            os.symlink("/dev/shm/" + shared.name, link)
            client = SlotClient(
                cfg,
                read_buffer=ctypes.addressof(
                    ctypes.c_ubyte.from_buffer(shared.mapping)
                ),
                read_buffer_size=shared.capacity,
                read_buffer_id=identity,
            )
            module = cls(ctx.engine_context, config, adapt(client), shared, link)
            module.client = client
            return module
        except BaseException:
            if client is not None:
                client.close()
            if link.is_symlink():
                link.unlink()
            shared.close()
            raise


class SlotPackedModule(Status, PackedCanonicalModule):
    @classmethod
    def create(cls, ctx: Any) -> Any:
        """Construct the configured adapter and its owned resources."""
        config = dict(ctx.config_for(cls))
        client = SlotClient(config["reader_adapter"])
        try:
            module = cls(ctx.engine_context, config, adapt(client))
        except BaseException:
            client.close()
            raise
        module.client = client
        return module
