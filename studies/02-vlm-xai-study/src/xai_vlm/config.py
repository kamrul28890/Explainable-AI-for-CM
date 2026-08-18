"""Settings that must not drift between models or stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = STUDY_ROOT / "results"
REPO_ROOT = STUDY_ROOT.parents[1]
MODEL_DIR = REPO_ROOT / "models"

# Qwen2.5-VL uses 14px patches merged 2x2 -> 28x28 pixel blocks. Vision-token
# count, and therefore KV-cache size, scales directly with pixel budget: on an
# 8 GB card resolution drives memory harder than model size does.
#
# THIS VALUE IS FIXED ACROSS EVERY MODEL AND STAGE. Varying it would confound the
# comparison with input size rather than model quality (architecture doc Risk #11).
PATCH_BLOCK = 28 * 28
MAX_PIXELS = 1280 * PATCH_BLOCK
MIN_PIXELS = 256 * PATCH_BLOCK

# Deterministic by default. Stability (metric 3) overrides these explicitly, which
# is the point of that metric -- with greedy decoding repeated runs would be
# byte-identical and score perfectly stable by construction.
GENERATION = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 128,
}

# COMPUTE DTYPE AND ATTENTION KERNEL ARE CORRECTNESS SETTINGS, NOT TUNING KNOBS.
#
# Qwen2.5-VL is trained in bfloat16. bf16 and fp16 are both 16 bits, but fp16
# spends them on precision rather than range, and large attention logits overflow
# it. Measured on three images, reason-first, 640-block budget:
#
#   fp16 + eager   ->  empty output; 128 special tokens, decodes to ''   0/3 correct
#   bf16 + eager   ->  text, but corrupted: spurious "addCriterion"
#                      tokens injected mid-sentence, "a a", "left left"   2/3 correct
#   bf16 + sdpa    ->  clean                                             3/3 correct
#
# So the two defects are independent: fp16 causes overflow, and the eager kernel
# corrupts the token stream even once bf16 has fixed the overflow. Run with fp16
# and a working model looks blind.
#
# ATTENTION_IMPL_FOR_READING is separate because reading attention (Sparsity,
# metric 2)
# requires attn_implementation="eager", which is the kernel that corrupts
# generation. The backend therefore never GENERATES under eager: it generates with
# sdpa and, where attention is needed, re-runs a single teacher-forced forward pass
# over the already-produced sequence.
COMPUTE_DTYPE = "bfloat16"
ATTENTION_IMPL = "sdpa"          # generation path
ATTENTION_IMPL_FOR_READING = "eager"   # single forward pass only, never generation


@dataclass(frozen=True)
class ModelSpec:
    """One model tier."""

    name: str
    repo_id: str
    quantization: str          # "4bit" | "8bit" | "none"
    supports_attention: bool   # can we read per-layer attention?
    notes: str = ""


MODEL_TIERS: dict[str, ModelSpec] = {
    "qwen7b-4bit": ModelSpec(
        name="qwen7b-4bit",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        quantization="4bit",
        supports_attention=True,
        notes="primary; largest that fits 8 GB",
    ),
    "qwen3b-8bit": ModelSpec(
        name="qwen3b-8bit",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        quantization="8bit",
        supports_attention=True,
        notes="fallback and size-comparison arm",
    ),
    "qwen3b-4bit": ModelSpec(
        name="qwen3b-4bit",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        quantization="4bit",
        supports_attention=True,
        notes="smoke tests when VRAM is tight",
    ),
}


def resolve_tier(name: str) -> ModelSpec:
    """Look up a tier, naming the valid options when it is unknown."""
    if name not in MODEL_TIERS:
        raise ValueError(
            f"unknown model tier {name!r}; choose from {sorted(MODEL_TIERS)}"
        )
    return MODEL_TIERS[name]
