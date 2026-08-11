"""Tests for model-reply parsing.

No GPU required -- these run anywhere, which makes prompt and parser work the
natural thing to do on the development machine.
"""

import pytest

from xai_vlm.parsing import ParseStats, parse_response

IMG = (1200, 900)


# --------------------------------------------------------------- happy path
def test_exact_format_is_strict():
    r = parse_response(
        "ANSWER: YES\n"
        "REGION: (410, 250, 520, 600)\n"
        "REASON: The worker on the left is not wearing a hard hat.",
        IMG,
    )
    assert r.ok and r.strict
    assert r.verdict == "YES"
    assert r.region == (410.0, 250.0, 520.0, 600.0)
    assert r.reason.startswith("The worker on the left")


def test_no_with_none_region_is_strict():
    r = parse_response("ANSWER: NO\nREGION: NONE\nREASON: All workers wear hard hats.", IMG)
    assert r.ok and r.strict
    assert r.verdict == "NO"
    assert r.region is None


# --------------------------------------------------------- tolerated variants
@pytest.mark.parametrize("text", [
    "answer: yes\nregion: (10, 10, 50, 50)\nreason: No helmet.",
    "**ANSWER:** YES\n**REGION:** (10, 10, 50, 50)\n**REASON:** No helmet.",
    "```\nANSWER: YES\nREGION: (10, 10, 50, 50)\nREASON: No helmet.\n```",
    "ANSWER - YES\nREGION - (10, 10, 50, 50)\nREASON - No helmet.",
])
def test_common_variations_still_parse(text):
    r = parse_response(text, IMG)
    assert r.ok, f"failed to parse: {text!r}"
    assert r.verdict == "YES"


def test_markdown_is_accepted_but_flagged_non_strict():
    # Cleaning changes the text, so this must not be silently counted as strict --
    # a rising non-strict rate is how prompt drift becomes visible.
    r = parse_response("**ANSWER:** YES\n**REGION:** NONE\n**REASON:** x", IMG)
    assert r.ok
    assert not r.strict


def test_bracketless_and_whitespace_separated_boxes():
    r = parse_response("ANSWER: YES\nREGION: 10 20 30 40\nREASON: x", IMG)
    assert r.region == (10.0, 20.0, 30.0, 40.0)


def test_inverted_coordinates_are_normalised_not_discarded():
    r = parse_response("ANSWER: YES\nREGION: (520, 600, 410, 250)\nREASON: x", IMG)
    assert r.region == (410.0, 250.0, 520.0, 600.0)
    assert not r.strict


def test_normalised_coordinates_are_converted_and_flagged():
    # Some models emit 0-1 despite being asked for pixels.
    r = parse_response("ANSWER: YES\nREGION: (0.1, 0.2, 0.3, 0.4)\nREASON: x", IMG)
    assert r.region == pytest.approx((120.0, 180.0, 360.0, 360.0))
    assert not r.strict


def test_out_of_bounds_box_is_clamped():
    r = parse_response("ANSWER: YES\nREGION: (-50, -50, 5000, 5000)\nREASON: x", IMG)
    assert r.region == (0.0, 0.0, 1200.0, 900.0)
    assert not r.strict


def test_true_false_accepted_as_yes_no():
    assert parse_response("ANSWER: TRUE\nREGION: NONE\nREASON: x", IMG).verdict == "YES"
    assert parse_response("ANSWER: FALSE\nREGION: NONE\nREASON: x", IMG).verdict == "NO"


def test_trailing_commentary_does_not_leak_into_reason():
    r = parse_response(
        "ANSWER: YES\nREGION: NONE\nREASON: No hard hat.\n\nLet me know if you need more.",
        IMG,
    )
    assert r.reason == "No hard hat."


# ------------------------------------------------------------------ failures
def test_empty_response_fails_cleanly():
    r = parse_response("", IMG)
    assert not r.ok
    assert r.error


def test_prose_without_a_verdict_fails():
    r = parse_response("This construction site looks quite dangerous overall.", IMG)
    assert not r.ok


def test_bare_yes_parses_but_is_never_strict():
    r = parse_response("YES, the worker has no hard hat.", IMG)
    assert r.ok and r.verdict == "YES"
    assert not r.strict


def test_degenerate_box_is_rejected_rather_than_kept():
    r = parse_response("ANSWER: YES\nREGION: (100, 100, 100, 100)\nREASON: x", IMG)
    assert r.verdict == "YES"
    assert r.region is None      # zero-area box is not a usable region
    assert not r.strict


# --------------------------------------------------------------------- stats
def test_stats_track_outcomes_and_parse_rate():
    stats = ParseStats()
    stats.record(parse_response("ANSWER: YES\nREGION: NONE\nREASON: a", IMG))
    stats.record(parse_response("**ANSWER:** NO\n**REGION:** NONE\n**REASON:** b", IMG))
    stats.record(parse_response("", IMG))

    assert stats.total == 3
    assert stats.strict == 1
    assert stats.lenient == 1
    assert stats.failed == 1
    assert stats.parse_rate == pytest.approx(2 / 3)


def test_stats_warn_below_95_percent():
    stats = ParseStats()
    for _ in range(9):
        stats.record(parse_response("ANSWER: YES\nREGION: NONE\nREASON: a", IMG))
    stats.record(parse_response("", IMG))
    assert "WARNING" in stats.summary()


def test_stats_do_not_warn_when_healthy():
    stats = ParseStats()
    for _ in range(100):
        stats.record(parse_response("ANSWER: YES\nREGION: NONE\nREASON: a", IMG))
    assert "WARNING" not in stats.summary()
