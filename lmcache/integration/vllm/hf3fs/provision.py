# SPDX-License-Identifier: Apache-2.0
"""Provision immutable slots under a caller-created 3FS directory."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import os
import uuid

# Local
from .slot_index import FILE_BYTES, LANES, PACK_BYTES, SLOT_BYTES, fdb


def provision(
    root: str, cluster_file: str, mount_point: str, files: int = 32
) -> dict[str, Any]:
    """Create sparse segments and publish their matching FDB layout.

    The caller must create ``root/.slots-v1`` with the desired 3FS chain table,
    chunk size and striping. Existing files are never truncated. Run one
    provisioner per namespace; runtime writers may start only after return.
    """
    root = str(Path(root).resolve())
    if (
        not Path(root).is_relative_to(Path(mount_point).resolve())
        or not 32 <= files <= 1024
        or files % LANES
    ):
        raise ValueError(
            "Root must be on the mount and file count a multiple of 32, <=1024"
        )
    target = Path(root) / ".slots-v1"
    if not target.is_dir():
        raise ValueError("Create the .slots-v1 directory with 3fs-admin first")
    prefix = b"LMCACHE-SLOTS-V1\x00" + hashlib.sha256(root.encode()).digest() + b"\x00"
    db = fdb.open(cluster_file)
    db.options.set_transaction_timeout(10000)
    manifest = target / "layout.json"
    existing = db[prefix + b"config"]
    if existing is not None:
        config = json.loads(existing)
        if json.loads(manifest.read_text()) != config or config["files"] != files:
            raise RuntimeError("Existing filesystem/index layout differs")
        for i in range(files):
            if (target / f"{i:016x}.bin").stat().st_size != FILE_BYTES:
                raise RuntimeError("Cannot repair published segments in place")
        return config

    def create(i: int) -> None:
        """Create sparse segment i, or verify an existing segment's exact size."""
        name = target / f"{i:016x}.bin"
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            if name.stat().st_size != FILE_BYTES:
                raise RuntimeError(
                    f"Unexpected existing segment size: {name}"
                ) from None
            return
        try:
            os.ftruncate(fd, FILE_BYTES)
        finally:
            os.close(fd)

    with ThreadPoolExecutor(16) as executor:
        list(executor.map(create, range(files)))
    config = dict(
        version=1,
        root=root,
        files=files,
        pack_bytes=PACK_BYTES,
        slot_bytes=SLOT_BYTES,
        file_bytes=FILE_BYTES,
        lanes=LANES,
        storage_chunk_bytes=1024**2,
        generation=uuid.uuid4().hex,
    )
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if {k: v for k, v in saved.items() if k != "generation"} != {
            k: v for k, v in config.items() if k != "generation"
        }:
            raise RuntimeError("Existing unpublished manifest differs")
        config = saved
    else:
        temporary = target / ("layout-" + config["generation"] + ".tmp")
        with temporary.open("x") as out:
            out.write(json.dumps(config, sort_keys=True))
            out.flush()
            os.fsync(out.fileno())
        temporary.rename(manifest)

    @fdb.transactional
    def publish(tr: Any) -> None:
        encoded = json.dumps(config, sort_keys=True).encode()
        old = tr[prefix + b"config"].value
        if old is not None and old != encoded:
            raise RuntimeError("Existing layout differs")
        tr[prefix + b"config"] = encoded

    publish(db)
    return config


def main() -> None:
    """Provision a configured namespace without changing any storage service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--cluster-file", required=True)
    parser.add_argument("--mount-point", required=True)
    parser.add_argument("--files", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(
            provision(args.root, args.cluster_file, args.mount_point, args.files)
        )
    )


if __name__ == "__main__":
    main()
