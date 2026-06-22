"""Visual sparsity: how concentrated is the cross-attention heatmap?

Computed only on the 24x24 visual-stream heatmap from attribution.py --
text-prompt-token attention lives in a different space (different sequence
length, mixed with location/label token semantics) and is not merged into
these numbers, per the pilot plan's design decision to report visual and
text attribution streams separately rather than as one combined score.
"""

import numpy as np


def topk_mass_ratio(attr_map: np.ndarray, k: int) -> float:
    """Fraction of total attribution mass held by the top-k highest cells.

    1.0 = maximally sparse (all mass concentrated in k cells); close to
    k/n = scattered (near-uniform attribution).
    """
    flat = attr_map.flatten().astype(float)
    total = flat.sum()
    if total <= 0:
        return float("nan")
    top_k_sum = np.sort(flat)[-k:].sum()
    return float(top_k_sum / total)


def regions_above_threshold(attr_map: np.ndarray, thresh: float) -> int:
    """Count of cells whose normalized attribution is >= thresh.

    Assumes attr_map is already normalized to [0, 1] (as
    attribution.cross_attention_heatmap returns it). Fewer cells above
    threshold means a more focused explanation.
    """
    return int((attr_map >= thresh).sum())
