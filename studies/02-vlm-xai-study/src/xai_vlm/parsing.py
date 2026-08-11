"""Parse a model's reply into a structured answer.

The required format is::

    ANSWER: YES
    REGION: (410, 250, 520, 600)
    REASON: The worker on the left is not wearing a hard hat.

Design stance: **be tolerant, but record that tolerance was needed.**

A strict parser silently discards replies that were perfectly understandable but
formatted loosely, and a study that quietly loses 15% of its data to formatting is
reporting a metric computed on an unrepresentative subset. An over-tolerant parser
has the opposite failure -- it invents structure that was not there.

So we accept common harmless variations (case, markdown emphasis, code fences,
extra prose around the block) and set ``strict=False`` whenever we had to. The
share of non-strict parses is reported alongside every metric, so a prompt that is
degrading can be spotted rather than averaged away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

Verdict = Literal["YES", "NO"]
Box = tuple[float, float, float, float]


@dataclass
class ParsedAnswer:
    """One parsed model reply."""

    verdict: Optional[Verdict] = None
    region: Optional[Box] = None
    reason: str = ""
    # False when the reply needed lenient handling to read. True means it matched
    # the requested format exactly.
    strict: bool = True
    # Populated when the reply could not be read at all. Such rows are excluded
    # from metrics and the exclusion is counted and reported -- never silent.
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is not None and not self.error

    @property
    def has_region(self) -> bool:
        return self.region is not None


# Markdown emphasis and code fences are the two things models add most often when
# asked for a plain-text block; stripping them is safe and loses no information.
_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", re.MULTILINE)
_EMPHASIS = re.compile(r"\*\*|__|`")

_ANSWER = re.compile(r"ANSWER\s*[:\-]\s*(YES|NO|TRUE|FALSE)\b", re.IGNORECASE)
_REGION = re.compile(r"REGION\s*[:\-]\s*(.+)", re.IGNORECASE)
_REASON = re.compile(r"REASON\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)

# Four numbers in any bracket style, comma or whitespace separated.
_FOUR_NUMBERS = re.compile(
    r"[\(\[\{]?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[\)\]\}]?"
)
_NONE = re.compile(r"\b(NONE|N/?A|NULL|NOT\s+APPLICABLE)\b", re.IGNORECASE)

_TRUTHY = {"YES": "YES", "TRUE": "YES", "NO": "NO", "FALSE": "NO"}


def _clean(text: str) -> tuple[str, bool]:
    """Strip fences and emphasis. Returns (cleaned, was_already_clean)."""
    cleaned = _EMPHASIS.sub("", _FENCE.sub("", text)).strip()
    return cleaned, cleaned == text.strip()


def _parse_region(raw: str, image_size: tuple[int, int] | None) -> tuple[Optional[Box], bool]:
    """Parse the REGION value. Returns (box, strict)."""
    if _NONE.search(raw):
        return None, True

    m = _FOUR_NUMBERS.search(raw)
    if not m:
        return None, False

    vals = [float(v) for v in m.groups()]
    strict = True

    # Some models emit normalised 0-1 coordinates despite being asked for pixels.
    # Detect and convert rather than discard -- but flag it, because a genuinely
    # tiny box in the top-left corner is indistinguishable from this and the
    # distinction matters.
    if image_size and all(0.0 <= v <= 1.0 for v in vals) and max(vals) <= 1.0:
        w, h = image_size
        vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
        strict = False

    x0, y0, x1, y1 = vals
    # Inverted coordinates are a common and harmless model slip; normalising is
    # safe because a box has no orientation.
    if x1 < x0:
        x0, x1 = x1, x0
        strict = False
    if y1 < y0:
        y0, y1 = y1, y0
        strict = False

    if x1 <= x0 or y1 <= y0:
        return None, False

    if image_size:
        w, h = image_size
        clamped = (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1))
        if clamped != (x0, y0, x1, y1):
            strict = False
        x0, y0, x1, y1 = clamped
        if x1 <= x0 or y1 <= y0:
            return None, False

    return (x0, y0, x1, y1), strict


def parse_response(text: str, image_size: tuple[int, int] | None = None) -> ParsedAnswer:
    """Parse one model reply.

    ``image_size`` enables clamping and normalised-coordinate detection. Without
    it, boxes are returned as given.
    """
    if not text or not text.strip():
        return ParsedAnswer(error="empty response")

    cleaned, was_clean = _clean(text)
    strict = was_clean

    m = _ANSWER.search(cleaned)
    if not m:
        # A bare yes/no with no ANSWER label is still usable, but well outside the
        # requested format, so it is never counted as strict.
        bare = re.match(r"\s*(YES|NO)\b", cleaned, re.IGNORECASE)
        if not bare:
            return ParsedAnswer(error="no ANSWER field", strict=False)
        verdict = _TRUTHY[bare.group(1).upper()]
        strict = False
    else:
        verdict = _TRUTHY[m.group(1).upper()]

    region: Optional[Box] = None
    m = _REGION.search(cleaned)
    if m:
        region, region_strict = _parse_region(m.group(1).splitlines()[0], image_size)
        strict = strict and region_strict
    else:
        strict = False

    reason = ""
    m = _REASON.search(cleaned)
    if m:
        # REASON is last in the format, so take the first line and stop at any
        # trailing commentary the model appended.
        reason = m.group(1).strip().splitlines()[0].strip()
    else:
        strict = False

    return ParsedAnswer(verdict=verdict, region=region, reason=reason, strict=strict)


@dataclass
class ParseStats:
    """Running tally of parse outcomes, reported with every metric."""

    total: int = 0
    strict: int = 0
    lenient: int = 0
    failed: int = 0
    retried: int = 0
    _errors: dict[str, int] = field(default_factory=dict)

    def record(self, parsed: ParsedAnswer, retried: bool = False) -> None:
        self.total += 1
        if retried:
            self.retried += 1
        if not parsed.ok:
            self.failed += 1
            self._errors[parsed.error] = self._errors.get(parsed.error, 0) + 1
        elif parsed.strict:
            self.strict += 1
        else:
            self.lenient += 1

    @property
    def parse_rate(self) -> float:
        """Share of replies that yielded a usable answer."""
        return self.ok_count / self.total if self.total else float("nan")

    @property
    def ok_count(self) -> int:
        return self.strict + self.lenient

    def summary(self) -> str:
        if not self.total:
            return "no responses parsed"
        lines = [
            f"parsed {self.ok_count}/{self.total} ({self.parse_rate:.1%})",
            f"  strict  {self.strict:>6}",
            f"  lenient {self.lenient:>6}",
            f"  failed  {self.failed:>6}",
            f"  retried {self.retried:>6}",
        ]
        for err, n in sorted(self._errors.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {n:>5}  {err}")
        if self.parse_rate < 0.95:
            lines.append("  WARNING: parse rate below 95% -- fix the prompt before "
                         "trusting any metric computed from this run")
        return "\n".join(lines)
