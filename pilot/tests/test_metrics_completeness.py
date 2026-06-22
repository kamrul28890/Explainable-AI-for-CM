from xai_pilot.metrics.completeness import classify_sample


def test_grid_fallback_is_no_usable_explanation_even_if_answer_changed():
    assert classify_sample("grid", answer_changed_top1=True, answer_changed_top2=True) == "no_usable_explanation"


def test_model_region_with_no_answer_change_is_weak():
    assert classify_sample("model", answer_changed_top1=False, answer_changed_top2=False) == "explanation_weak"


def test_model_region_with_top1_change_is_supported():
    assert classify_sample("model", answer_changed_top1=True, answer_changed_top2=False) == "explanation_supported"


def test_model_region_with_only_top2_change_is_supported():
    assert classify_sample("model", answer_changed_top1=False, answer_changed_top2=True) == "explanation_supported"
