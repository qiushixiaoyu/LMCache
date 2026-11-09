Canonical pack storage on HF3FS
===============================

This opt-in module consolidates the DeepSeek V4.1 canonical-pack CPU storage
path. It is separate from the normal ``hf3fs`` L2 configuration and from
SGLang's HF3FS page backend. Importing LMCache does not activate it.

The native connector includes bounded multi-object reads, optional registered
FD leases, sharded legacy files and pre-created large-file offsets. The default
native options remain ``immutable_fd_capacity=0`` and ``hash_shards=False``.
The canonical module explicitly enables these options on an immutable namespace.

Layout and completion
---------------------

Each complete 128-token chunk contains eight canonical components from source
layers 2, 8, 14 and 20. A pack is 224 KiB, including its 256-byte header. Each
pack occupies an append-only 256 KiB slot in a sparse 4 GiB file. FoundationDB
maps the immutable pack key to the committed slot. Thirty-two allocation lanes
spread consecutive slots across files. The native reader retains registered
FDs and issues reads at their offsets; it can also read the earlier per-object
flat and hash-sharded layouts.

A new write reserves unique slots, completes native data I/O, and then commits
the FDB mapping. Repeated writes of an acknowledged key do not rewrite its data.
Concurrent duplicate writers allocate distinct slots; the first published
mapping wins. Failed or abandoned reservations never authorize slot reuse.
There is no durable deletion, eviction, compaction or slot reclamation here.
Keep the FDB index and filesystem manifest with their segment files.

The shared reader registers a node-local pool with 3FS and reads missing packs
into their final pool slots. Consumers receive offsets over a Unix socket.
Singleflight joins concurrent requests for the same pack. A 32-pack window
returns all eight components in one descriptor request. Leases last until every
consumer copy has completed, and can be released in batches. No consumer
liveness scan or per-request ``/proc`` identity read reclaims memory. A crashed
consumer's lease remains pinned until a drained pool rebuild.

Configuration and provisioning
------------------------------

Build LMCache's CPU extensions with the HF3FS SDK available; see :doc:`hf3fs`.
The canonical index additionally requires the FoundationDB Python binding supporting API version 730 and a compatible FDB
client library. The existing cluster must already
provide its FUSE mount and ``3fs-virt/iovs`` registration directory. This module
does not change storage replication, checksums, NIC configuration or quotas.

Create an isolated root and ``.slots-v1`` directory using ``3fs-admin`` with the
desired chain table and striping before provisioning. For the existing six-disk
layout, use chain table 1, a 1 MiB storage chunk and stripe size 18. Then run::

    python -m lmcache.integration.vllm.hf3fs.provision \
      --root /mnt/3fs/canonical-test \
      --mount-point /mnt/3fs \
      --cluster-file /etc/foundationdb/fdb.cluster \
      --files 32

Run one provisioner per namespace. Existing segments are checked, never
truncated. Thirty-two sparse files provide 128 GiB of slot address space;
this is not physical preallocation. The original storage capacity guard must
continue to account for actual bytes and replicas.

Add the following option to an otherwise configured CPU MP server::

    --server-module '{
      "module_path": "lmcache.integration.vllm.hf3fs.prefill_module",
      "config": {
        "concurrency": 4,
        "shm_slots": 131072,
        "socket_path": "/tmp/lmcache-hf3fs-read.sock",
        "verify_pack_hashes": false,
        "reader_adapter": {
          "mount_point": "/mnt/3fs",
          "base_paths": "/mnt/3fs/canonical-test",
          "cluster_file": "/etc/foundationdb/fdb.cluster",
          "num_workers": 8,
          "store_workers": 32,
          "ior_entries": 128,
          "immutable_fd_capacity": 8192
        }
      }
    }'

Use ``decode_module`` for the byte-returning module without a shared pool.
The default shared pool is 131072 packs / 28 GiB. Set a smaller explicit
``shm_slots`` for CPU correctness tests. The shared module skips optional
layout/identity/body hashes when ``verify_pack_hashes=false`` but retains magic
and size checks. Writers still construct the original versioned header.

The public MP extension methods are ``canonical.put``, ``canonical.exists``
and ``canonical.get``; their MessagePack payload is a list of component
``[key, byte_size]`` descriptors, with a third bytes field for PUT. PUT requires
all eight components and accepts at most 32 complete packs per call. Call them
through ``RequestClientFactory`` / ``ServerModuleCallRequest``. The shared
module additionally provides ``canonical.get_shm`` and ``canonical.release_shm``.
``SharedClient`` uses the configured local Unix socket for these operations.

Scope and recovery
------------------

* This branch contains the selected synchronous completion path with the
  short-lock/no-process-polling pool. The asynchronous-Future/tile-size tuning
  candidates that did not establish a gain are not enabled or copied in.
* ``prefix_progress`` retains bounded positive durable-prefix knowledge.
  Only contiguous acknowledged intervals may prove prior chunks. A tail object
  alone cannot prove an earlier gap caused by failed or incremental writes.
* ``window_reader.load`` retains the two-window prefetch/source-layer protocol.
  ``hf3fs_copy`` is an optional CUDA batch-copy helper in the CUDA platform.
  The caller must prove copy completion before releasing its session. This
  does not implement RNIC/GPUDirect RDMA loading.
* Existing vLLM scheduler/model hooks, HTTP routing, P/D transport, DSpark and
  server launchers are external to this module. No inference service is
  restarted or automatically converted by this branch.
* FD and positive-index caches assume immutable keys and no external deletion.
  Drain and restart clients before deleting/rebuilding the namespace. Switching
  new writes to ``reader_adapter.legacy_writes=true`` still requires a reader
  that understands existing slots; an old flat-only reader is not a rollback.

Validation
----------

The CPU native tests use a deterministic SDK double and ASAN/UBSAN. They cover
partial and out-of-order completions, failed preparation/submission, cache
leases and slot offsets. Python tests cover singleflight, cancellation,
read failure/retry, prefix gaps and window ownership. From the repository root::

    python -m pytest --noconftest -q tests/integration/vllm/hf3fs

``tests/integration/vllm/hf3fs/cluster_roundtrip.py`` additionally accepts an
explicit disposable, provisioned namespace and verifies the actual native
connector, MP module APIs, shared descriptors and cross-host CPU bytes. It
creates no model and allocates no GPU memory. GPU/model performance is a
separate acceptance step; historical bandwidth is not a new measurement of
this consolidated branch.
