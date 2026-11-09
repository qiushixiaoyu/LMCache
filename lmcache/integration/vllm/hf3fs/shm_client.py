# SPDX-License-Identifier: Apache-2.0
"""Read-only shared views and explicit retrieval completion for GPU consumers."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import mmap
import os
import re
import socket
import time
import urllib.request
import uuid

# Third Party
import msgpack

# First Party
from lmcache.v1.platform.devices.cuda.hf3fs_copy import register, unregister

# Local
from .shm_pool import process_cookie
from .shm_socket import read_frame, write_frame

# Keep an explicit bound; the COMM5 pool uses 28 GiB (131072 complete packs).
MAX_POOL_BYTES = 32 * 1024**3


class SharedClient:
    def __init__(
        self, port: int = 18181, socket_path: str = "/tmp/lmcache-hf3fs-read.sock"
    ) -> None:
        self.port = port
        self.socket_path = socket_path
        self.pid = os.getpid()
        self.cookie = process_cookie(self.pid)
        self.registered = 0
        self.pin = False
        self.before_unmap = lambda mapping: None
        self.mapping = None
        self.name = None
        self.lease = None
        self.counts = {}
        self.channel = "http"
        self.connection = None
        self.pending_releases = []
        self.defer_release = False
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, operation: str, value: Any) -> Any:
        """Send one bounded descriptor operation over the configured local transport."""
        if self.channel == "unix":
            if self.connection is None:
                self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.connection.settimeout(35)
                self.connection.connect(self.socket_path)
            try:
                write_frame(self.connection, [operation, value])
                response = read_frame(self.connection)
            except (OSError, EOFError, ValueError, msgpack.UnpackException):
                self.connection.close()
                self.connection = None
                raise
            if not response["ok"]:
                raise RuntimeError(response["error"])
            return msgpack.unpackb(response["payload"], raw=False), len(
                response["payload"]
            )
        encoded = msgpack.packb(value, use_bin_type=True)
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/{operation}",
            data=encoded,
            headers={"Content-Type": "application/msgpack"},
        )
        with self.opener.open(request, timeout=35) as response:
            raw = response.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("Oversized metadata response")
        return msgpack.unpackb(raw, raw=False), len(raw)

    def get(self, rows: list[list[Any]]) -> Any:
        """Obtain pack views; the consumer must release them after its final copy
        completes.
        """
        if self.lease is not None:
            raise RuntimeError("Previous shared lease still active")
        self.lease = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response, nbytes = self.call(
                "get_shm",
                dict(
                    rows=rows,
                    lease=self.lease,
                    pid=self.pid,
                    cookie=self.cookie,
                    releases=self.pending_releases,
                ),
            )
            self.pending_releases.clear()
            received = time.perf_counter()
            if (
                response["version"] != 1
                or response["lease"] != self.lease
                or response["ready"] is not True
                or len(response["items"]) != len(rows)
                or not re.fullmatch("lc3fs_read_[0-9a-f]{32}", response["pool"])
                or type(response["pool_bytes"]) is not int
                or not 0 < response["pool_bytes"] <= MAX_POOL_BYTES
            ):
                raise ValueError("Invalid shared response")
            name = response["pool"]
            if self.name != name:
                if self.mapping is not None:
                    self.before_unmap(self.mapping)
                    if self.registered:
                        unregister(self.registered)
                        self.registered = 0
                    self.mapping.close()
                fd = os.open("/dev/shm/" + name, os.O_RDWR | os.O_NOFOLLOW)
                try:
                    if os.fstat(fd).st_size != response["pool_bytes"]:
                        raise ValueError("Shared pool size mismatch")
                    self.mapping = mmap.mmap(
                        fd, response["pool_bytes"], access=mmap.ACCESS_WRITE
                    )
                finally:
                    os.close(fd)
                self.name = name
            if self.pin and not self.registered:
                self.registered = register(self.mapping)
            values = []
            for item, row in zip(response["items"], rows, strict=True):
                if item is None:
                    values.append(None)
                    continue
                offset, size, generation = item
                if (
                    any(type(x) is not int for x in item)
                    or size != row[1]
                    or offset < 0
                    or offset + size > response["pool_bytes"]
                    or generation <= 0
                ):
                    raise ValueError("Out-of-bounds shared descriptor")
                values.append(memoryview(self.mapping)[offset : offset + size])
            self.counts = dict(
                remote_wait_seconds=received - started,
                shm_map_validate_seconds=time.perf_counter() - received,
                shm_metadata_bytes=nbytes,
                shm_kv_payload_bytes=0,
            )
            return values
        except BaseException:
            if self.connection is not None:
                self.connection.close()
                self.connection = None
            self.release(force=True)
            raise

    def release(self, force: bool = False) -> Any:
        """Release a completed lease, or record cancellation before admission."""
        if self.lease is not None:
            self.pending_releases.append(self.lease)
            self.lease = None
        if not self.pending_releases or (self.defer_release and not force):
            return 0.0
        started = time.perf_counter()
        value = (
            {"leases": self.pending_releases}
            if self.channel == "unix"
            else {"lease": self.pending_releases[0]}
        )
        self.call("release_shm", value)
        self.pending_releases.clear()
        return time.perf_counter() - started

    def configure(self, channel: str) -> None:
        """Switch transport only after releasing the current lease and connection."""
        self.release(force=True)
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.channel = channel
        self.defer_release = channel == "unix"

    def close(self) -> None:
        """Drain owned work before releasing its resources."""
        self.release(force=True)
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.mapping is not None:
            self.before_unmap(self.mapping)
            if self.registered:
                unregister(self.registered)
                self.registered = 0
            self.mapping.close()
            self.mapping = None
