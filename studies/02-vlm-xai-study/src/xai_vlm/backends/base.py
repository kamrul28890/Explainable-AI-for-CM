"""The Backend interface every model must satisfy.

This is the load-bearing abstraction of the study. Every metric is written **once**,
against this interface, and each model supplies one implementation.

Why it matters: if each model had its own metric code, a difference in results
could be a difference in the *measuring instrument* rather than in the model. One
shared harness makes that impossible by construction, which is what licenses the
comparison in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from xai_vlm.parsing import Box, ParsedAnswer, Verdict
from xai_vlm.rules import RuleId


@dataclass
class Answer:
    """One model reply, normalised across backends.

    Every metric consumes this and nothing else, so a metric can never accidentally
    depend on a model-specific detail.
    """

    verdict: Optional[Verdict]
    region: Optional[Box] = None
    rationale: str = ""
    confidence: float = float("nan")
    raw_text: str = ""
    latency_ms: float = 0.0
    parse_strict: bool = True
    parse_error: str = ""
    # Set when this answer came from an intervention rather than a clean run, so
    # provenance is never lost between stages.
    intervention: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict is not None and not self.parse_error

    @classmethod
    def from_parsed(cls, parsed: ParsedAnswer, **kw) -> "Answer":
        return cls(
            verdict=parsed.verdict,
            region=parsed.region,
            rationale=parsed.reason,
            parse_strict=parsed.strict,
            parse_error=parsed.error,
            **kw,
        )


@runtime_checkable
class Backend(Protocol):
    """What every model implementation must provide."""

    name: str

    def answer(self, image: Image.Image, rule_id: RuleId, **kw) -> Answer:
        """Ask one safety rule about one image."""
        ...

    def answer_masked(
        self, image: Image.Image, rule_id: RuleId, region: Box, method: str = "black", **kw
    ) -> Answer:
        """Ask again with `region` removed.

        `method` selects the intervention (architecture doc section 2.7):

        ``"black"``   patch-aligned black fill -- comparable with prior work, but
                      creates a rectangle no real photograph contains, so a flip may
                      mean "this image looks broken" rather than "the evidence is gone".
        ``"tokens"``  delete the image tokens for those patches before the language
                      model runs -- no out-of-distribution pixels exist, because no
                      image is modified. This is the primary, artefact-free measure.
        ``"inpaint"`` reconstruct the region from its surroundings.
        ``"mean"``    fill with the image mean, closer to the natural distribution
                      than black.

        Reporting ``black`` and ``tokens`` side by side measures how much the
        black-rectangle artefact inflates the flip rate.
        """
        ...

    def attention(self, image: Image.Image, rule_id: RuleId, **kw) -> Optional[np.ndarray]:
        """Attention over image patches as a 2-D grid, or None if unavailable.

        Returning None is legitimate -- not every backend exposes attention -- and
        Sparsity is then reported only for the backends where it works, rather than
        being silently recorded as zero.
        """
        ...

    def patch_grid(self) -> tuple[int, int]:
        """(rows, cols) of the vision encoder's patch grid.

        Needed to snap mask boundaries outward to patch borders: a mask cutting a
        patch in half leaves a partial, ambiguous signal in it.
        """
        ...


class BackendUnavailable(RuntimeError):
    """Raised when a backend cannot run here.

    Carries the reason so a stage script can skip and report rather than crash --
    e.g. bitsandbytes quantisation requested on Apple Silicon, where it has no build.
    """
