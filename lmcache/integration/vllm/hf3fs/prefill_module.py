# SPDX-License-Identifier: Apache-2.0
"""Discoverable canonical HF3FS server module."""

# Future
from __future__ import annotations

# Local
from .slot_module import SlotSharedModule


class PrefillModule(SlotSharedModule):
    """Opt-in canonical pack service; configured through --server-module."""
