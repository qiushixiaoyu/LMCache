# SPDX-License-Identifier: Apache-2.0
"""Bounded positive knowledge for an immutable, non-evicting cache namespace.

This is an acceleration index, not a source of durability. Only successful
storage acknowledgments / positive existence replies may populate it. It is
discarded on restart, and LRU eviction only costs a later storage query.
"""

# Future
from __future__ import annotations

# Standard
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any
import threading


class PrefixIndex:
    """A chain-hash endpoint proves the entire preceding prefix, not HBM state."""

    def __init__(
        self, chunk_tokens: int, hash_tokens: int, capacity: int = 65536
    ) -> None:
        if (
            hash_tokens <= 0
            or chunk_tokens <= 0
            or chunk_tokens % hash_tokens
            or capacity <= 0
        ):
            raise ValueError("Invalid prefix geometry or capacity")
        self.chunk = chunk_tokens
        self.hash_tokens = hash_tokens
        self.capacity = capacity
        self.confirmed: OrderedDict[tuple[int, bytes], None] = OrderedDict()

    def key(self, hashes: list[bytes], end: int) -> Any:
        """Encode an immutable identity for the corresponding persistence index."""
        return end, bytes(hashes[end // self.hash_tokens - 1])

    def longest(self, hashes: list[bytes], end: int, floor: int = 0) -> int:
        """Find the longest confirmed chain endpoint within the requested complete-
        chunk range.
        """
        end = min(end, len(hashes) * self.hash_tokens) // self.chunk * self.chunk
        for pos in range(end, floor, -self.chunk):
            key = self.key(hashes, pos)
            if key in self.confirmed:
                self.confirmed.move_to_end(key)
                return pos
        return floor

    def confirm(self, hashes: list[bytes], start: int, end: int) -> bool:
        """Record a successful contiguous prefix; refuse to bridge an unproven gap."""
        if (
            not (0 <= start <= end <= len(hashes) * self.hash_tokens)
            or start % self.chunk
            or end % self.chunk
        ):
            raise ValueError("Invalid durable interval")
        # A later successful range cannot bridge a failed or unknown earlier one.
        if start and self.key(hashes, start) not in self.confirmed:
            return False
        for pos in range(start + self.chunk, end + 1, self.chunk):
            key = self.key(hashes, pos)
            self.confirmed[key] = None
            self.confirmed.move_to_end(key)
            if len(self.confirmed) > self.capacity:
                self.confirmed.popitem(last=False)
        return True


class PositiveObjects:
    """Thread-safe LRU of exact component keys; never cache a miss or failure."""

    def __init__(self, capacity: int = 262144) -> None:
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        self.capacity = capacity
        self.known: OrderedDict[str, None] = OrderedDict()
        self.lock = threading.Lock()

    def note(self, keys: Iterable[str]) -> None:
        """Remember acknowledged positive identities within the LRU capacity."""
        with self.lock:
            for key in keys:
                self.known[key] = None
                self.known.move_to_end(key)
                if len(self.known) > self.capacity:
                    self.known.popitem(last=False)

    def exists(self, keys: list[str], query: Any) -> Any:
        """Query unknown identities and cache only successful positive replies."""
        with self.lock:
            cached = [key in self.known for key in keys]
            for key, hit in zip(keys, cached, strict=True):
                if hit:
                    self.known.move_to_end(key)
        missing = list(
            dict.fromkeys(key for key, hit in zip(keys, cached, strict=True) if not hit)
        )
        values = query(missing) if missing else []
        if len(values) != len(missing) or any(value < 0 for value in values):
            raise RuntimeError("Invalid or failed durable existence response")
        self.note(key for key, value in zip(missing, values, strict=True) if value == 1)
        replies = dict(zip(missing, values, strict=True))
        return [
            1 if hit else replies[key] for key, hit in zip(keys, cached, strict=True)
        ], sum(cached)
