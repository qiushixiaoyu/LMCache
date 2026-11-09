# SPDX-License-Identifier: Apache-2.0
"""Keep the current HF3FS backend; combine old release and new reservation."""

# Future
from __future__ import annotations

# Third Party
import msgpack

# First Party
from lmcache.v1.multiprocess.ext_server_module import server_module_handler

# Local
from .pack_format import checked_groups
from .shm_module import SharedCanonicalModule


class LockSharedModule(SharedCanonicalModule):
    @server_module_handler("canonical.release_shm")
    def release_shm(self, payload: bytes) -> bytes:
        if len(payload) > 8192:
            raise ValueError("Oversized release")
        request = msgpack.unpackb(payload, raw=False)
        tokens = request.get("leases", [request.get("lease")])
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= 128
            or any(not isinstance(t, str) or len(t) != 32 for t in tokens)
        ):
            raise ValueError("Invalid leases")
        self.shared.release_many(tokens)
        return msgpack.packb({"released": True}, use_bin_type=True)

    @server_module_handler("canonical.get_shm")
    def get_shm(self, payload: bytes) -> bytes:
        if len(payload) > 1024**2:
            raise ValueError("Oversized descriptor request")
        request = msgpack.unpackb(payload, raw=False)
        rows = request["rows"]
        if not isinstance(rows, list) or not 0 < len(rows) <= 4096:
            raise ValueError("Invalid bounded descriptor batch")
        for offset in range(0, len(rows), 256):
            checked_groups("get", rows[offset : offset + 256])
        if sum(r[1] for r in rows) > 64 * 1024**2:
            raise ValueError("Logical batch byte budget exceeded")
        result = self.shared.acquire(
            rows,
            request["lease"],
            request["pid"],
            request["cookie"],
            releases=request.get("releases", ()),
        )
        encoded = msgpack.packb(result, use_bin_type=True)
        with self.lock:
            self.counts["shm_rpc_metadata_bytes"] += len(encoded)
            self.counts["shm_rpc_kv_payload_bytes"] += 0
        return encoded
