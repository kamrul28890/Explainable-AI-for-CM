"""Visual attribution: where did the model attend while grounding a phrase?

Florence-2 has no ViT-style, self-attention-only encoder to run textbook
"attention rollout" (Abnar & Zuidema 2020) over. Its encoder runs
self-attention over a *fused* sequence: `[1 global pooled image token] +
[576 image-patch tokens, 24x24] + [text prompt tokens]`. This was verified
directly against the cached `modeling_florence2.py`, not assumed:
`_encode_image` returns `torch.cat([spatial_avg_pool_x, temporal_avg_pool_x])`
(1 + 576 = 577 tokens), and 576 = 24x24 follows from the DaViT vision
tower's four conv stages (strides 4,2,2,2 with the given kernel/padding)
applied to the processor's fixed 768x768 input. Rolling out *that*
self-attention stack would mix text-token attention into the result in a
way the original method was never defined for, and risks a silent grid
misalignment (577 is not a perfect square -- the global token at index 0
has to be dropped before reshaping the remaining 576 into 24x24, or every
heatmap is off by one cell).

The architecturally clean signal available here instead is the decoder's
cross-attention back to the encoder sequence, computed fresh per generated
output token -- "where did the model look to produce this token," with no
rollout approximation needed. `cross_attention_heatmap` averages that
cross-attention over decoder layers, heads, and generated tokens, then
keeps only the 576 image-patch positions.

Generation here always uses `num_beams=1` (greedy), unlike the `num_beams=3`
used for the actual answers in Days 3-5: HF's beam search reorders the
batch dimension at every step, and `generate()`'s returned attentions are
not automatically un-reordered, so beam-search cross-attentions don't line
up sample-by-sample without extra bookkeeping this pilot doesn't need. The
box this heatmap nominally explains is therefore a fresh greedy decode of
the same prompt, not the original beam-search answer from earlier days --
checked against it in day6_findings.md.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

# Verified against the model's actual encoder layout (see module docstring):
# 768x768 input / vision-tower stride 32 = 24x24 patches; index 0 in the
# image-feature block is the pooled "spatial_avg_pool" token, not a patch.
PATCH_GRID = (24, 24)
N_PATCH_TOKENS = PATCH_GRID[0] * PATCH_GRID[1]
GLOBAL_TOKEN_OFFSET = 1


@dataclass
class AttributionResult:
    """Cross-attention artifact and metadata from one greedy grounding run."""

    heatmap: np.ndarray  # (24, 24), normalized to [0, 1]
    greedy_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    n_generated_tokens: int = 0


def cross_attention_heatmap(
    model,
    processor,
    image: Image.Image,
    task_token: str,
    text_input: str | None = None,
    max_new_tokens: int = 1024,
) -> AttributionResult:
    """Decoder->encoder cross-attention over image patches, averaged over layers/heads/steps."""
    prompt = task_token if text_input is None else task_token + text_input
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

    # Attribution is observational: gradients are unnecessary, and disabling
    # them reduces memory pressure during generation on the pilot GPU.
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    if not out.cross_attentions:
        return AttributionResult(heatmap=np.zeros(PATCH_GRID), greedy_boxes=[], n_generated_tokens=0)

    # `generate` returns one tuple per generated token and one tensor per
    # decoder layer. Average layers, batch, heads, and target-token position
    # within each step, then average the remaining source attention over all
    # generated steps. The final vector has one value per encoder token.
    step_means = []
    for step_layers in out.cross_attentions:
        layer_stack = torch.stack(list(step_layers))  # (n_layers, 1, heads, 1, src_len)
        step_means.append(layer_stack.mean(dim=(0, 1, 2, 3)))  # (src_len,)
    seq_mean = torch.stack(step_means).mean(dim=0)  # (src_len,)

    # Drop the pooled global image token and ignore prompt-text tokens after
    # the 576 visual patches before reshaping to the known 24x24 grid.
    image_attn = seq_mean[GLOBAL_TOKEN_OFFSET : GLOBAL_TOKEN_OFFSET + N_PATCH_TOKENS]
    heatmap = image_attn.reshape(PATCH_GRID).float().cpu().numpy()
    # Min-max normalization supports visual overlays and threshold metrics.
    # A constant map stays all-zero to avoid division by zero.
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    raw_text = processor.batch_decode(out.sequences, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(raw_text, task=task_token, image_size=(image.width, image.height))
    boxes = [tuple(b) for b in parsed.get(task_token, {}).get("bboxes", [])]

    return AttributionResult(heatmap=heatmap, greedy_boxes=boxes, n_generated_tokens=len(out.cross_attentions))
