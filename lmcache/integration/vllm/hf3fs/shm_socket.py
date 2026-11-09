# SPDX-License-Identifier: Apache-2.0
"""Node-local descriptor socket: one hop to the MP-owned read pool."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import os
import socket
import socketserver
import struct
import threading

# Third Party
import msgpack

PATH = "/tmp/lmcache-hf3fs-read.sock"
LIMIT = 1024**2


def receive(sock: Any, size: int) -> bytes:
    """Receive exactly the bounded frame length or report a closed connection."""
    parts = bytearray()
    while len(parts) < size:
        data = sock.recv(size - len(parts))
        if not data:
            raise EOFError("Descriptor connection closed")
        parts.extend(data)
    return bytes(parts)


def read_frame(sock: Any) -> Any:
    """Decode one size-bounded MessagePack frame from the Unix connection."""
    size = struct.unpack("!I", receive(sock, 4))[0]
    if not 0 < size <= LIMIT:
        raise ValueError("Invalid descriptor frame size")
    return msgpack.unpackb(receive(sock, size), raw=False)


def write_frame(sock: Any, data: Any) -> None:
    """Send a bounded MessagePack frame with a network-order length prefix."""
    raw = msgpack.packb(data, use_bin_type=True)
    if not 0 < len(raw) <= LIMIT:
        raise ValueError("Oversized descriptor frame")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, module: Any, path: Any = PATH) -> None:
        self.module = module
        self.slots = threading.BoundedSemaphore(32)
        self.path = Path(path)
        if self.path.exists():
            raise RuntimeError(
                "Previous descriptor socket must be drained before reuse"
            )
        super().__init__(str(self.path), Handler)
        os.chmod(self.path, 0o660)
        self.thread = threading.Thread(target=self.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        """Drain owned work before releasing buffers, handles and completion resources."""
        self.shutdown()
        self.server_close()
        self.path.unlink(missing_ok=True)


class Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        """Serve descriptor operations after checking the Unix peer process identity."""
        if not self.server.slots.acquire(blocking=False):
            return
        try:
            pid, uid, gid = struct.unpack(
                "3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
            )
            while True:
                try:
                    operation, value = read_frame(self.request)
                except (EOFError, ConnectionError):
                    return
                try:
                    if operation == "get_shm":
                        if value["pid"] != pid:
                            raise ValueError("Peer PID mismatch")
                        payload = self.server.module.get_shm(
                            msgpack.packb(value, use_bin_type=True)
                        )
                    elif operation == "release_shm":
                        payload = self.server.module.release_shm(
                            msgpack.packb(value, use_bin_type=True)
                        )
                    else:
                        raise ValueError("Unknown descriptor operation")
                    write_frame(self.request, {"ok": True, "payload": payload})
                except (ValueError, RuntimeError, TimeoutError, KeyError) as error:
                    write_frame(self.request, {"ok": False, "error": str(error)})
        finally:
            self.server.slots.release()
