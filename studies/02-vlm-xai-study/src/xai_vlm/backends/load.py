"""One way to get a backend for a tier.

Frozen tiers get the frozen QwenBackend unless XAI_BACKEND=hf, so their results
reproduce byte-for-byte by default. Every other tier gets the shared HFVLMBackend.
"""
from __future__ import annotations

from xai_vlm import config


def backend_class(tier: str):
    config.resolve_tier(tier)          # an unknown tier fails before any import
    if config.uses_legacy_backend(tier):
        from xai_vlm.backends.qwen import QwenBackend
        return QwenBackend
    from xai_vlm.backends.hf import HFVLMBackend
    return HFVLMBackend


def load_backend(tier: str, **kw):
    return backend_class(tier)(tier=tier, **kw)
