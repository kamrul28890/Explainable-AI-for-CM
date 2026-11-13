"""The embedding boundary. The real model is exercised in one marked test only."""
import numpy as np
import pytest

from xai_vlm.metrics.text_similarity import FakeEmbedder, cosine


def test_cosine_of_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_orthogonal_vectors_is_zero():
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_of_zero_vector_is_zero_not_nan():
    """An empty rationale must score 0, not poison the arithmetic with nan."""
    assert cosine(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


def test_fake_embedder_returns_scripted_vectors():
    fake = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    out = fake.encode(["a", "b"])
    assert out.shape[0] == 2
    assert cosine(out[0], out[1]) == pytest.approx(0.0)


def test_fake_embedder_unknown_text_is_orthogonal_to_everything():
    """Unscripted text must not accidentally score as similar."""
    fake = FakeEmbedder({"a": [1.0, 0.0, 0.0]})
    out = fake.encode(["a", "never seen"])
    assert cosine(out[0], out[1]) == pytest.approx(0.0)
