"""Contract every Backend must satisfy, exercised against FakeBackend.

Running these against a fake is deliberate: it verifies the harness contract
without a GPU, and the same tests are re-run against QwenBackend in Task 7.
"""
import numpy as np
import pytest
from PIL import Image

from xai_vlm.backends import Answer, Backend
from xai_vlm.backends.fake import FakeBackend


@pytest.fixture
def img():
    return Image.new("RGB", (1200, 900), (128, 128, 128))


@pytest.fixture
def backend():
    return FakeBackend()


def test_satisfies_the_protocol(backend):
    assert isinstance(backend, Backend)


def test_answer_returns_a_usable_Answer(backend, img):
    a = backend.answer(img, "rule_1")
    assert isinstance(a, Answer)
    assert a.verdict in {"YES", "NO"}
    assert a.ok
    assert a.latency_ms >= 0


def test_scripted_replies_are_returned_in_order(img):
    b = FakeBackend(script={"rule_1": [
        "ANSWER: YES\nREGION: (10, 20, 110, 220)\nREASON: no hard hat",
        "ANSWER: NO\nREGION: NONE\nREASON: all wearing hats",
    ]})
    first, second = b.answer(img, "rule_1"), b.answer(img, "rule_1")
    assert first.verdict == "YES"
    assert first.region == (10.0, 20.0, 110.0, 220.0)
    assert second.verdict == "NO"
    assert second.region is None


def test_masked_answer_records_the_intervention(backend, img):
    a = backend.answer_masked(img, "rule_1", (10, 10, 100, 100), method="tokens")
    assert a.intervention == "tokens"


def test_every_call_is_recorded_for_assertions(backend, img):
    backend.answer(img, "rule_1")
    backend.answer_masked(img, "rule_2", (0, 0, 10, 10), method="black")
    assert [c[0] for c in backend.calls] == ["answer", "answer_masked"]


def test_attention_is_a_normalised_2d_grid(backend, img):
    att = backend.attention(img, "rule_1")
    assert isinstance(att, np.ndarray)
    assert att.ndim == 2
    assert att.min() >= 0.0 and att.max() <= 1.0


def test_patch_grid_is_two_positive_ints(backend):
    rows, cols = backend.patch_grid()
    assert rows > 0 and cols > 0


def test_unparseable_reply_yields_a_not_ok_answer(img):
    b = FakeBackend(script={"rule_1": ["I cannot tell from this image."]})
    a = b.answer(img, "rule_1")
    assert not a.ok
    assert a.parse_error
