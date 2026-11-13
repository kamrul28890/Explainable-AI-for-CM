"""Settings that must not drift between models or stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[2]
# Every stage reads and writes here. XAI_RESULTS_DIR redirects a whole run to a
# separate tree -- e.g. a run on the RCAC Scholar cluster -- so it can never
# resume into, or overwrite, the frozen local results. Unset, it is unchanged.
# Replies are not reproducible across GPUs (see runner.backend_device), so a
# second machine needs its own tree, not its own rows in this one.
_RESULTS_OVERRIDE = os.environ.get("XAI_RESULTS_DIR")
RESULTS_DIR = Path(_RESULTS_OVERRIDE) if _RESULTS_OVERRIDE else STUDY_ROOT / "results"
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
# WHICH FRAME A MODEL'S REGION COORDINATES ARE READ IN (found 2026-11-12, see
# xai_vlm.frames). "legacy" reads them as original-image pixels -- the frozen
# behaviour, kept so every frozen file reproduces byte-for-byte. "rescaled"
# reads them in the resized model-input frame and maps them to the original,
# which is what the model actually means. A rescaled run must write to its own
# results tree (XAI_RESULTS_DIR); its fingerprint records the flag.
REGION_FRAME = os.environ.get("XAI_REGION_FRAME") or "legacy"  # empty means unset
if REGION_FRAME not in ("legacy", "rescaled"):
    raise ValueError(f"XAI_REGION_FRAME must be 'legacy' or 'rescaled', not {REGION_FRAME!r}")


def region_frame_fingerprint() -> dict:
    """Empty under the frozen default, so existing run fingerprints still match."""
    return {} if REGION_FRAME == "legacy" else {"region_frame": REGION_FRAME}


# WHICH BACKEND CLASS RUNS A TIER (roster step 2, 2026-11-13). "legacy" keeps the
# frozen QwenBackend for the tiers that have frozen results, so they reproduce
# byte-for-byte. "hf" runs them through the shared HFVLMBackend instead -- used to
# prove the two agree, see scripts/20_backend_parity.py. Tiers with no frozen
# results always take the shared backend.
BACKEND = os.environ.get("XAI_BACKEND") or "legacy"  # empty means unset
if BACKEND not in ("legacy", "hf"):
    raise ValueError(f"XAI_BACKEND must be 'legacy' or 'hf', not {BACKEND!r}")


def uses_legacy_backend(tier: str) -> bool:
    return BACKEND == "legacy" and resolve_tier(tier).legacy_region_frame


def backend_fingerprint(tier: str) -> dict:
    """Empty for a frozen tier on the frozen class, so its fingerprints still match.
    Anything else records the class and the coordinate convention it was read in.
    A name outside MODEL_TIERS is a test double (the scripts' --model only accepts
    real tiers) and contributes nothing.

    The transformers version is recorded whenever it is not environment A's, so
    rows from the two environments (roster D5) can never resume into one file."""
    fp: dict = {}
    if tier in MODEL_TIERS and not uses_legacy_backend(tier):
        fp = {"backend": "hf", "box_convention": resolve_tier(tier).box_convention}
    v = transformers_version()
    if v != FROZEN_TRANSFORMERS:
        fp["transformers"] = v
    return fp


# Environment A, in which every frozen result was produced.
FROZEN_TRANSFORMERS = "4.49.0"


def transformers_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("transformers")
    except PackageNotFoundError:
        return ""


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
    # Folder under MODEL_DIR holding the weights; empty means the tier's own name.
    # Quantisation is applied at load time, so tiers of one checkpoint share it.
    weights: str = ""
    # Which adapter in backends.families builds inputs, reads the token grid and
    # merges vision features for this model.
    family: str = "qwen2_5_vl"
    # The unit the model states REGION in (frames.CONVENTIONS). It sets both the
    # REGION line of the prompt and how the reply is mapped to original pixels.
    box_convention: str = "input_px"
    # "A" is the frozen transformers 4.49 environment, "B" the current release.
    env: str = "A"
    # True only for the tiers whose frozen results were read in the original
    # frame; they follow XAI_REGION_FRAME. Every other tier is always mapped.
    legacy_region_frame: bool = False

    @property
    def weights_dir(self) -> str:
        return self.weights or self.name


MODEL_TIERS: dict[str, ModelSpec] = {
    "qwen7b-4bit": ModelSpec(
        name="qwen7b-4bit",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        quantization="4bit",
        supports_attention=True,
        notes="primary; largest that fits 8 GB",
        legacy_region_frame=True,
    ),
    "qwen7b-bf16": ModelSpec(
        name="qwen7b-bf16",
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        quantization="none",
        supports_attention=True,
        notes="unquantised comparison arm, A40 only (~16 GB); what nf4 costs",
        weights="qwen7b-4bit",
        legacy_region_frame=True,
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
    # --- model roster (docs/superpowers/specs/2026-11-12-study2-model-roster-design.md)
    "qwen3b-bf16": ModelSpec(
        name="qwen3b-bf16",
        repo_id="Qwen/Qwen2.5-VL-3B-Instruct",
        quantization="none",
        supports_attention=True,
        notes="roster 3: size, down",
    ),
    "qwen32b-4bit": ModelSpec(
        name="qwen32b-4bit",
        repo_id="Qwen/Qwen2.5-VL-32B-Instruct",
        quantization="4bit",
        supports_attention=True,
        notes="roster 4: size, up; nf4 (~20 GB) on one A40",
        weights="qwen32b",
    ),
    "qwen32b-bf16": ModelSpec(
        name="qwen32b-bf16",
        repo_id="Qwen/Qwen2.5-VL-32B-Instruct",
        quantization="none",
        supports_attention=True,
        notes="roster 4 at full precision: ~66 GB, split over two A40s (NGPU=2)",
        weights="qwen32b",
    ),
}


def resolve_tier(name: str) -> ModelSpec:
    """Look up a tier, naming the valid options when it is unknown."""
    if name not in MODEL_TIERS:
        raise ValueError(
            f"unknown model tier {name!r}; choose from {sorted(MODEL_TIERS)}"
        )
    return MODEL_TIERS[name]
