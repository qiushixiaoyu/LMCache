# SPDX-License-Identifier: Apache-2.0
"""Verify the deliberate integrity-policy change and retained structural limits."""

# Standard
from unittest.mock import patch
import hashlib
import json

# First Party
from lmcache.integration.vllm.hf3fs.pack_format import (
    BODY_SIZE,
    COMPONENTS,
    HEADER,
    descriptor,
    pack,
)
from lmcache.integration.vllm.hf3fs.pack_validation import valid_pack

GROUP = (
    "dsv41_canonical_v1@"
    + hashlib.sha256(b"pack-hash-policy-20261102").hexdigest()
    + "@"
    + "a" * 64
)
ROWS = [
    descriptor(GROUP, c) + [bytes([i + 1]) * descriptor(GROUP, c)[1]]
    for i, c in enumerate(COMPONENTS)
]


def test_validation_policy():
    raw = pack(GROUP, ROWS)
    assert valid_pack(GROUP, raw, True) and valid_pack(GROUP, raw, False)
    # Changed layout/identity/digest/body each fails full checks but is accepted
    # with the explicit no-hash policy. No production object is modified.
    for offset in (8, 40, 72, HEADER.size + 17):
        altered = bytearray(raw)
        altered[offset] ^= 1
        assert not valid_pack(GROUP, altered, True)
        assert valid_pack(GROUP, altered, False)
    assert not valid_pack(GROUP[:-1] + "b", raw, True)
    assert valid_pack(GROUP[:-1] + "b", raw, False)
    header = list(HEADER.unpack_from(raw))
    invalid = [raw[:-1], raw + b"x"]
    for index, value in ((0, b"BADMAGIC"), (4, BODY_SIZE - 1)):
        changed = header.copy()
        changed[index] = value
        invalid.append(HEADER.pack(*changed) + raw[HEADER.size :])
    for data in invalid:
        assert not valid_pack(GROUP, data, True)
        assert not valid_pack(GROUP, data, False)
    # The disabled branch must not invoke even a small identity SHA.
    with patch(
        "lmcache.integration.vllm.hf3fs.pack_validation.hashlib.sha256",
        side_effect=AssertionError("hash invoked"),
    ):
        assert valid_pack(GROUP, raw, False)
    print(
        json.dumps(
            dict(
                event="validation_policy",
                status="PASS",
                hash_calls_when_disabled=0,
                structural_failures_rejected=len(invalid),
                hash_corruption_cases=5,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    test_validation_policy()
