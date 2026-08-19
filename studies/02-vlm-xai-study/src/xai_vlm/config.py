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

# WHICH SPLITS THE STUDY DRAWS FROM. Lives here because Stage 1 (sampling) and
# Stage 2 (inference) must agree, and when they disagree nothing detects it.
#
# They did disagree once: Stage 1 was changed to scan both splits while Stage 2
# still streamed only `test`. Every train-split image was silently never found,
# its rows never ran, and the run exited 0 with a plausible summary computed on a
# third of the manifest. Guards cannot catch that -- they only see rows that were
# attempted. Hence one constant, imported by both.
#
# Both splits, not `test` alone: the model is evaluated zero-shot and never
# trained on any of this data, so there is nothing to hold out, and `test` alone
# puts rule_2 at n=25 and rule_4 at n=24, under the MIN_N_FLOOR of 30.
SPLITS: tuple[str, ...] = ("test", "train")

# 640 blocks, not 1280. Measured on 8 violation + 8 compliant images, bf16+sdpa,
# reason-first:
#
#   blocks   grid      tokens   mean latency   peak VRAM   sensitivity
#     1280   30x41       1230        112.3 s     9.5 GB     8/8 = 100%
#      640   21x29        609          3.5 s     7.0 GB     8/8 = 100%
#
# Identical sensitivity at 32x the speed. 1280 exceeds the card and spills into
# Windows' shared-memory fallback, which trades an OOM for a silent ~30x
# slowdown; 640 fits. On the five compliant images both levels saw, specificity
# was 2/5 vs 1/5 -- a one-image difference, noise at that n.
#
# This retires the small-object risk of architecture doc 7.4 for rule_1: the
# hazards the model catches at full resolution it also catches at 640. It is what
# makes the study feasible -- roughly 27 h for the full battery instead of 31 days.
MAX_PIXELS = 640 * PATCH_BLOCK
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
