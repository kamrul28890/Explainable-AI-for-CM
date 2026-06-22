from xai_pilot.inference import AnswerResult
from xai_pilot.metrics.stability import answer_agreement_rate, region_overlap_score
from xai_pilot.regions import Region


def test_answer_agreement_rate_all_same_is_one():
    results = [AnswerResult(answer="compliant") for _ in range(3)]
    assert answer_agreement_rate(results) == 1.0


def test_answer_agreement_rate_all_different_is_zero():
    results = [
        AnswerResult(answer="compliant"),
        AnswerResult(answer="violation"),
        AnswerResult(answer="violation"),
    ]
    # pairs: (compliant,violation)=disagree, (compliant,violation)=disagree, (violation,violation)=agree
    assert abs(answer_agreement_rate(results) - 1 / 3) < 1e-9


def test_answer_agreement_rate_single_result_is_nan():
    import math

    assert math.isnan(answer_agreement_rate([AnswerResult(answer="compliant")]))


def test_region_overlap_score_identical_boxes_is_one():
    regions = [Region(box=(0, 0, 10, 10), source="model") for _ in range(3)]
    assert region_overlap_score(regions) == 1.0


def test_region_overlap_score_disjoint_boxes_is_zero():
    regions = [
        Region(box=(0, 0, 10, 10), source="model"),
        Region(box=(100, 100, 110, 110), source="model"),
    ]
    assert region_overlap_score(regions) == 0.0


def test_region_overlap_score_nan_when_any_run_used_grid_fallback():
    import math

    regions = [
        Region(box=(0, 0, 10, 10), source="model"),
        Region(box=(0, 0, 25, 20), source="grid", label="grid"),
    ]
    assert math.isnan(region_overlap_score(regions))
