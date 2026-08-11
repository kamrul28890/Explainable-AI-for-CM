"""CI guard (Scale-up Phase 1.7): known-magic literals must not reappear in code.

The pilot shipped a bug where sample counts and class sizes were hardcoded as
literals into analysis/plotting code (e.g. n=163, the 768->24 patch-grid math),
which silently produced wrong labels when the data changed. Every such value
must be read from data or config instead. This test tokenizes the pipeline
source and fails if a forbidden numeric literal appears as actual code (NUMBER
tokens only, so comments and docstrings that legitimately *describe* the pilot's
163 samples or the 24x24 grid are not flagged).
"""

import tokenize
from pathlib import Path

PILOT_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [PILOT_ROOT / "scripts", PILOT_ROOT / "src" / "xai_pilot"]

# Dataset-derived counts and patch-grid dimensions that must never be typed as
# literals in code -- they belong to the data (sample counts) or are derived at
# runtime from the actual input resolution (patch grid). See attribution._patch_grid.
FORBIDDEN_NUMBERS = {163, 3004, 576, 768}


def _python_files():
    for base in SCAN_DIRS:
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def _forbidden_number_hits(path: Path):
    hits = []
    with open(path, "rb") as f:
        try:
            tokens = list(tokenize.tokenize(f.readline))
        except tokenize.TokenError:
            return hits
    for tok in tokens:
        if tok.type == tokenize.NUMBER:
            try:
                value = int(tok.string)
            except ValueError:
                continue  # floats / non-int literals
            if value in FORBIDDEN_NUMBERS:
                hits.append((tok.start[0], tok.string))
    return hits


def test_no_forbidden_magic_number_literals_in_pipeline_code():
    offenders = []
    for path in _python_files():
        for line_no, literal in _forbidden_number_hits(path):
            rel = path.relative_to(PILOT_ROOT)
            offenders.append(f"{rel}:{line_no}: forbidden literal {literal}")
    assert not offenders, (
        "Hardcoded magic literals found (read from data/config instead):\n"
        + "\n".join(offenders)
    )
