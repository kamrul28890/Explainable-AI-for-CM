"""A Backend that returns scripted replies, for testing without a GPU.

Every stage is written against the Backend protocol, so the whole harness --
sampling, masking bookkeeping, metric aggregation, CSV writing -- can be built and
tested on any machine. Only the real model calls need CUDA.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from xai_vlm.backends.base import Answer
from xai_vlm.parsing import Box, parse_response
from xai_vlm.rules import RuleId

DEFAULT_REPLY = "ANSWER: YES\nREGION: (100, 100, 300, 400)\nREASON: A worker has no hard hat."


class FakeBackend:
    """Deterministic stand-in for a real model.

    `script` maps a rule id to a list of raw reply strings, consumed in order and
    then repeating the last one. Anything not scripted returns DEFAULT_REPLY.
    """

    name = "fake"

    def __init__(self, script: Optional[dict[str, list[str]]] = None,
                 patch_grid_size: tuple[int, int] = (32, 32)):
        self._script = {k: list(v) for k, v in (script or {}).items()}
        self._cursor: dict[str, int] = {}
        self._grid = patch_grid_size
        self.calls: list[tuple] = []

    def _next_reply(self, rule_id: str) -> str:
        replies = self._script.get(rule_id)
        if not replies:
            return DEFAULT_REPLY
        i = self._cursor.get(rule_id, 0)
        self._cursor[rule_id] = i + 1
        return replies[min(i, len(replies) - 1)]

    def answer(self, image: Image.Image, rule_id: RuleId, **kw) -> Answer:
        self.calls.append(("answer", rule_id, kw))
        raw = self._next_reply(rule_id)
        parsed = parse_response(raw, image.size)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=1.0)

    def answer_masked(self, image: Image.Image, rule_id: RuleId, region: Box,
                      method: str = "black", **kw) -> Answer:
        self.calls.append(("answer_masked", rule_id, method, region))
        raw = self._next_reply(rule_id)
        parsed = parse_response(raw, image.size)
        return Answer.from_parsed(parsed, raw_text=raw, latency_ms=1.0,
                                  intervention=method)

    def attention(self, image: Image.Image, rule_id: RuleId, **kw) -> np.ndarray:
        self.calls.append(("attention", rule_id))
        rows, cols = self._grid
        # A deterministic peaked map, so tests can assert on concentration.
        grid = np.zeros((rows, cols), dtype=float)
        grid[rows // 2, cols // 2] = 1.0
        grid[rows // 2 - 1: rows // 2 + 2, cols // 2 - 1: cols // 2 + 2] += 0.25
        return np.clip(grid / grid.max(), 0.0, 1.0)

    def patch_grid(self, image: Optional[Image.Image] = None) -> tuple[int, int]:
        # The fake has a fixed grid by construction, so the image is ignored --
        # but the signature must match the protocol or harness code written
        # against a real backend cannot be exercised against this one.
        return self._grid
