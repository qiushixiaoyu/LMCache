# Canonical HF3FS integration

This package is specific to the existing DeepSeek V4.1 canonical layout. Its
model-dependent names and geometry belong under `integration/vllm`, not in
LMCache's generic distributed cache API. The native connector only understands
byte buffers and the versioned internal `SLOT1@file@offset` coordinate encoding.

The branch consolidates the selected metadata, slot-store, shared-lock,
merged-window and no-process-polling implementations. It does not import a
SGLang branch or change the 3FS server. The normal HF3FS L2 factory keeps its
existing defaults; the canonical server modules are explicitly configured.

## Ownership contracts

1. `SlotClient` retains submitted memoryviews until its native completion broker
   resolves the task. Store completion additionally waits for FDB publication.
2. Registered-file cache entries are shared leases. Eviction removes a cache
   reference; it does not close a descriptor still used by native I/O.
3. A pool entry cannot be reused while a lease holds it or native loading is
   incomplete. Request timeout is not I/O cancellation. Failed loads become
   retryable only after references and I/O are drained.
4. `Windows` owns the descriptors for every source layer. A failed CUDA
   completion retains the session; the caller's established worker retirement
   mechanism must terminate the poisoned worker. A deadline or a dead PID
   cannot establish DMA completion.
5. Durable prefix knowledge requires contiguous confirmed intervals. CPU pool
   cache eviction is independent of durable storage: this slot store has no
   durable deletion or reuse.

## Packaging changes from the experiments

All imports are package-relative or public LMCache APIs. There is no `sys.path`
overlay, dated `/data03` dependency or runtime source replacement. The selected
pool is a single class rather than dynamically importing the historical base.
FDB cluster and mount paths are explicit. Modules retain their MessagePack
operations and 224 KiB pack format. The optional CUDA helper lives in the CUDA
platform and performs no device work on import.

The integrated CPU API and ownership tests must pass on an isolated namespace.
These tests do not certify the external custom vLLM hooks or recreate the
historical 4.449 GB/s read / 0.963 GB/s write benchmark.
