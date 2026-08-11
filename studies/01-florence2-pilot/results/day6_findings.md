# Day 6 Findings: Visual Sparsity

## Method deviation from the plan (verified, not assumed)

The plan called for `attention_rollout()` first. Before writing it, I checked
Florence-2's actual cached modeling code
(`modeling_florence2.py`) rather than assume a textbook ViT-style rollout
would apply, and found it doesn't cleanly:

- Florence-2's encoder runs **one shared self-attention stack over a fused
  image+text sequence**: `[1 global "spatial_avg_pool" token] + [576
  image-patch tokens, 24x24] + [text prompt tokens]`. Verified directly --
  `_encode_image` returns `torch.cat([spatial_avg_pool_x, temporal_avg_pool_x])`
  (1 + 576 = 577 tokens, confirmed by printing `image_features.shape`), and
  576 = 24x24 follows from the DaViT vision tower's four conv stages
  (strides 4,2,2,2, kernel/padding per `ConvEmbed`) applied to the
  processor's fixed 768x768 input (computed by hand: 768->192->96->48->24).
- Classic attention rollout (Abnar & Zuidema 2020) is defined for a
  self-attention-only stack processing one modality. Rolling out Florence-2's
  encoder would recursively mix in attention to text tokens at every layer,
  which the method was never designed for, and isn't separable back out
  after the fact.
- **Used decoder->encoder cross-attention instead**: for each generated
  output token, Florence-2's decoder attends back to the full encoder
  sequence. This is well-defined per generation step with no rollout
  approximation needed. `attribution.cross_attention_heatmap` averages it
  over decoder layers, heads, and generated tokens, then drops the global
  token (index 0) before reshaping the remaining 576 values into the 24x24
  patch grid. The off-by-one here was a real trap: 577 is not a perfect
  square, and assuming `sqrt(num_image_tokens)` without checking would have
  silently misaligned every heatmap by one cell.
- **Decoding mode differs from Days 3-5**: cross-attention extraction always
  uses `num_beams=1` (greedy), not the `num_beams=3` beam search used for
  the actual logged answers. HF's beam search reorders the batch dimension
  every step, and `generate()`'s returned attentions aren't un-reordered
  automatically -- untangling that wasn't worth it for a pilot. Spot-checked
  on `0000007`: the greedy boxes (worker box, hard-hat box) matched Day 4's
  beam-search boxes closely in position, so this substitution doesn't appear
  to change *what* gets attributed, only confirms the attention is clean to
  read.

## Results

Ran `scripts/06_visual_sparsity.py` over all 163 samples: for each, computed
the cross-attention heatmap for whichever phrase produced Day 4's top-ranked
region (worker phrase, or the rule's object phrase), then scored
concentration with `topk_mass_ratio` and `regions_above_threshold`. 4/163
samples (Day 4's grid-fallback cases, no model box at all) were excluded --
no grounded phrase exists to attribute attention to.

| Metric | Mean (159 scored samples) |
|---|---|
| Top-5-of-576-cells mass ratio | 0.108 |
| Top-20-of-576-cells mass ratio | 0.247 |
| Cells above 0.5 (of 576) | 7.0 |

By `assigned_rule_id` (topk5_mass_ratio): rule_1 0.125, rule_2 0.113,
rule_3 0.098, rule_4 0.079.

By `attributed_phrase` (topk5_mass_ratio): **hard hat 0.195**, safety
harness 0.118, worker 0.115, guardrail 0.099, **excavator 0.079**.

## Finding: concentration is driven mainly by the target's physical size, not just "explanation quality"

`hard hat` is roughly 2.5x more concentrated than `excavator`. Checked
whether this tracks the grounded object's own box size (mean box area per
phrase, from `baseline_predictions.csv`):

| Phrase | Mean box area (px²) | topk5_mass_ratio |
|---|---|---|
| excavator | 287,913 | 0.079 |
| guardrail | 310,291 | 0.099 |
| safety harness | 178,491 | 0.118 |
| worker | 79,005 | 0.115 |
| hard hat | 94,330 | 0.195 |

Correlation between log(box area) and `topk5_mass_ratio` across all scored
samples: **-0.70**. Visually confirmed on two saved overlays:
`results/figures/sparsity/0000023_rule_1_hard_hat.png` shows a small, sharp
hotspot exactly on a distant worker's hard hat; `0000079_rule_3_guardrail.png`
shows attention spread broadly across most of the frame for a guardrail
spanning much of the image width. `0000007_rule_1_worker.png` (worker
phrase) vs. the hard-hat phrase on the same image makes the same point at
the single-image level: the hard-hat heatmap sits tightly on the yellow
helmet, the worker heatmap spreads loosely across his head and torso.

This is a real confound, not a bug: a small object's 576-cell patch grid
necessarily has fewer cells it *could* cover, so its attention mass is
mechanically concentrated into fewer cells regardless of whether the
underlying reasoning is any good. **Visual sparsity, as measured here,
partly reflects target footprint size rather than purely reflecting
explanation focus.** Worth flagging explicitly in Day 11's analysis --
comparing sparsity across rules without controlling for object size (rule 4
attributes to "excavator," a large object, vs. rule 1 to "hard hat," a small
one) will make rule 4's explanations look artificially worse than rule 1's
by this metric alone.

## Visual confirmation

- `0000023_rule_1_hard_hat.png`: small, sharp hotspot on a distant worker
  wearing (presumably) a hard hat, in an otherwise empty dirt lot -- the
  clearest "focused" example in the sample.
- `0000079_rule_3_guardrail.png`: attention spread broadly across most of
  the frame, consistent with the guardrail's large, elongated structural
  footprint -- the clearest "scattered" example.
- `0000007_rule_1_worker.png` and the hard-hat phrase on the same image
  (generated during the sanity check, not saved as part of the 10-sample
  visualization set but reproducible via `attribution.cross_attention_heatmap`):
  same image, two phrases, two different concentration levels -- isolates
  the phrase/target-size effect from any image-level confound.
