# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-cluster CPU checks; provision and remove the namespace separately."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import select
import tempfile
import time

# Third Party
import msgpack

# First Party
from lmcache.integration.vllm.hf3fs.pack_format import COMPONENTS, descriptor, pack
from lmcache.integration.vllm.hf3fs.shm_client import SharedClient
from lmcache.integration.vllm.hf3fs.slot_client import SlotClient
from lmcache.integration.vllm.hf3fs.slot_index import PACK_BYTES
from lmcache.integration.vllm.hf3fs.slot_module import (
    SlotPackedModule,
    SlotSharedModule,
)


def group(i: int) -> str:
    """Return a stable test identity shared by both cluster clients."""
    return "dsv41_canonical_v1@" + "e" * 64 + "@" + f"{i:064x}"


def rows(i: int, put: bool = False) -> list[list]:
    """Create complete model-shaped synthetic bytes outside model execution."""
    result = [descriptor(group(i), component) for component in COMPONENTS]
    if put:
        for n, row in enumerate(result):
            row.append(hashlib.shake_256(f"{i}/{n}".encode()).digest(row[1]))
    return result


def one(
    client: SlotClient, op: str, keys: list[str], buffers: list | None = None
) -> list[bool]:
    """Wait for public native task completion with its buffers retained."""
    submit = getattr(client, "submit_batch_" + op)
    fid = submit(keys) if buffers is None else submit(keys, buffers)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        select.select([client.event_fd()], [], [], 0.1)
        completed = client.drain_completions()
        if completed:
            assert len(completed) == 1 and completed[0][0] == fid
            _, ok, error, bits = completed[0]
            assert ok, error
            return bits
    raise RuntimeError("Cluster test timed out; keep process alive for native drain")


def run(cfg: dict, mode: str, start: int) -> None:
    """Exercise module RPCs, shared descriptors and cross-client persistence."""
    ids = list(range(start, start + 40))
    module_cfg = dict(
        reader_adapter=cfg, concurrency=4, shm_slots=64, verify_pack_hashes=False
    )
    ctx = SimpleNamespace(engine_context=None, config_for=lambda cls: module_cfg)
    writer = SlotPackedModule.create(ctx)
    try:
        if mode == "write":
            for offset in range(0, len(ids), 16):
                batch = [r for i in ids[offset : offset + 16] for r in rows(i, True)]
                assert msgpack.unpackb(writer.put(msgpack.packb(batch))) == [0] * len(
                    batch
                )
        for i in ids:
            assert msgpack.unpackb(writer.exists(msgpack.packb(rows(i)))) == [1] * 8
            assert msgpack.unpackb(writer.get(msgpack.packb(rows(i)))) == [
                r[2] for r in rows(i, True)
            ]
        # Duplicate writes must keep the original published location/data.
        if mode == "write":
            original = writer.client.index.lookup(
                [
                    f"{k.model_name}@{k.kv_rank:08x}@{k.object_group_id:x}@{k.chunk_hash.hex()}"
                    for k in writer.keys([group(ids[0])])
                ]
            )
            assert original[0] is not None
            writer.put(msgpack.packb(rows(ids[0], True)))
            assert (
                writer.client.index.lookup(
                    [
                        f"{k.model_name}@{k.kv_rank:08x}@{k.object_group_id:x}@{k.chunk_hash.hex()}"
                        for k in writer.keys([group(ids[0])])
                    ]
                )
                == original
            )
    finally:
        writer.close()
    with tempfile.TemporaryDirectory(prefix="lc3fs-socket-") as tmp:
        module_cfg["socket_path"] = str(Path(tmp) / "read.sock")
        reader = SlotSharedModule.create(ctx)
        consumer = SharedClient(socket_path=module_cfg["socket_path"])
        consumer.configure("unix")
        try:
            for offset in range(0, len(ids), 16):
                expected = [r for i in ids[offset : offset + 16] for r in rows(i, True)]
                values = consumer.get([r[:2] for r in expected])
                assert [bytes(v) for v in values] == [r[2] for r in expected]
                for value in values:
                    value.release()
                consumer.release(force=True)
            assert reader.report_status()["canonical_objects"]["shared"]["leases"] == 0
        finally:
            consumer.close()
            reader.close()
    # Mixed misses and backwards-compatible per-object files on the same namespace.
    key = "compat@00000001@0@" + hashlib.sha256(str(start).encode()).hexdigest()
    payload = pack(group(start), rows(start, True))
    if mode == "write":
        legacy = SlotClient(dict(cfg, legacy_writes=True))
        try:
            assert one(legacy, "set", [key], [memoryview(payload)]) == [True]
        finally:
            legacy.close()
    fresh = SlotClient(cfg)
    try:
        buffers = [memoryview(bytearray(PACK_BYTES)) for _ in range(2)]
        assert one(fresh, "get", [key, key + "missing"], buffers) == [True, False]
        assert bytes(buffers[0]) == payload
    finally:
        fresh.close()
    print(json.dumps(dict(mode=mode, start=start, packs=len(ids), passed=True)))


def main() -> None:
    """Run only on an explicitly provided disposable cluster namespace."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--mount-point", required=True)
    p.add_argument("--cluster-file", required=True)
    p.add_argument("--mode", choices=["write", "read"], required=True)
    p.add_argument("--start", type=int, required=True)
    args = p.parse_args()
    run(
        dict(
            mount_point=args.mount_point,
            base_paths=args.root,
            cluster_file=args.cluster_file,
            ior_entries=128,
        ),
        args.mode,
        args.start,
    )


if __name__ == "__main__":
    main()
