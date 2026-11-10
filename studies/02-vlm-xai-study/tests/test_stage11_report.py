"""The report collector, over hand-written CSVs. No GPU.

`test_reporting.py` covers the bootstrap itself. This covers the layer above it:
which rates get collected, and whether the sections they are filed under survive
into a table a paper can \\input.
"""
import csv
import importlib.util
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "stage11", STUDY_ROOT / "scripts" / "11_report.py")
stage11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage11)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _da_row(image_id, rule_id, verdict, flip_black, flip_tokens):
    return {
        "image_id": image_id, "rule_id": rule_id, "baseline_verdict": verdict,
        "region": "(1.0, 1.0, 2.0, 2.0)", "region_snapped": "False",
        "verdict_black": "NO", "flip_black": str(flip_black),
        "latency_ms_black": "1000",
        "verdict_tokens": "NO", "flip_tokens": str(flip_tokens),
        "latency_ms_tokens": "1000",
        "methods_disagree": str(flip_black != flip_tokens), "error": "",
    }


def _descriptive_accuracy_csv(tmp_path: Path) -> Path:
    """40 YES rows per rule and 40 NO controls -- all above the n=30 floor."""
    rows = []
    for rule in ("rule_1", "rule_4"):
        for i in range(40):
            # rule_1 flips on 4 of 40; rule_4 on 20 of 40. A per-rule breakdown
            # must be able to tell those two apart.
            flips = i < (4 if rule == "rule_1" else 20)
            rows.append(_da_row(f"{rule}-{i:03d}", rule, "YES", flips, flips))
    for i in range(40):
        rows.append(_da_row(f"ctrl-{i:03d}", "rule_1", "NO", i < 20, i < 20))
    _write(tmp_path / "descriptive_accuracy_m.csv", rows)
    return tmp_path


def _sections(rows):
    return [r["section"] for r in rows]


def test_each_section_is_one_contiguous_block(tmp_path):
    """render_latex opens a new tabular every time the section changes, so a
    section that appears, stops and reappears becomes several half-tables with
    duplicate headers instead of the one table the paper inputs."""
    res = _descriptive_accuracy_csv(tmp_path)
    sections = _sections(stage11.collect(res, "m", n_boot=200))

    first_seen = []
    for s in sections:
        if s not in first_seen:
            first_seen.append(s)
    runs = [s for i, s in enumerate(sections) if i == 0 or sections[i - 1] != s]
    assert runs == first_seen, f"section reappears after being left: {runs}"


def test_both_masking_methods_share_one_descriptive_accuracy_section(tmp_path):
    res = _descriptive_accuracy_csv(tmp_path)
    rows = stage11.collect(res, "m", n_boot=200)
    labels = [r["label"] for r in rows
              if r["section"] == "Stage 4 - descriptive accuracy"]
    assert "flip rate (black)" in labels
    assert "flip rate (tokens)" in labels


def test_flip_rate_is_reported_per_rule(tmp_path):
    """Pooling hides the subgroup that matters: the baseline found the model
    nearly blind to rule_4, and a pooled flip rate is dominated by rule_1."""
    res = _descriptive_accuracy_csv(tmp_path)
    rows = stage11.collect(res, "m", n_boot=200)
    by_rule = {r["label"]: r for r in rows
               if r["section"] == "Stage 4 - flip rate by rule (black)"}
    assert set(by_rule) == {"rule_1", "rule_4"}
    assert by_rule["rule_1"]["point"] == 4 / 40
    assert by_rule["rule_4"]["point"] == 20 / 40
    assert by_rule["rule_1"]["n"] == 40


def test_per_rule_flip_rate_counts_only_yes_rows(tmp_path):
    """The 40 NO controls are rule_1 rows too. Counting them into rule_1's flip
    rate would mix the artefact floor into the signal it is measured against."""
    res = _descriptive_accuracy_csv(tmp_path)
    rows = stage11.collect(res, "m", n_boot=200)
    rule_1 = next(r for r in rows
                  if r["section"] == "Stage 4 - flip rate by rule (black)"
                  and r["label"] == "rule_1")
    assert rule_1["n"] == 40


def test_method_disagreement_is_reported(tmp_path):
    """Two maskings of the same region disagreeing is evidence about the
    masking, not the model, and belongs in the artefact section."""
    res = _descriptive_accuracy_csv(tmp_path)
    rows = stage11.collect(res, "m", n_boot=200)
    labels = [r["label"] for r in rows if r["section"] == "Stage 4 - artefact floor"]
    assert "methods disagree" in labels


def test_missing_result_files_collect_nothing(tmp_path):
    assert stage11.collect(tmp_path, "absent", n_boot=200) == []


def _sparsity_csv(tmp_path):
    """40 rows per rule. rule_1 beats its own area null on 30 of 40; rule_4 on 10."""
    rows = []
    for rule, wins in (("rule_1", 30), ("rule_4", 10)):
        for i in range(40):
            beat = i < wins
            rows.append({
                "image_id": f"{rule}-{i:03d}", "rule_id": rule, "verdict": "YES",
                "attention_available": "True", "gini": "0.55",
                "mass_inside_region": "0.04" if beat else "0.01",
                "region_area_fraction": "0.02",
                "latency_ms": "20000", "error": "",
            })
    # A row where attention never came back must not count as a miss.
    rows.append({
        "image_id": "none-001", "rule_id": "rule_1", "verdict": "YES",
        "attention_available": "False", "gini": "", "mass_inside_region": "",
        "region_area_fraction": "", "latency_ms": "20000", "error": "",
    })
    _write(tmp_path / "sparsity_m.csv", rows)
    return tmp_path


def test_sparsity_reports_attention_beating_the_area_null(tmp_path):
    """The headline is not the raw mass -- it is mass against the area the region
    occupies, which is what uniform attention would deliver by chance."""
    res = _sparsity_csv(tmp_path)
    rows = stage11.collect(res, "m", n_boot=200)
    row = next(r for r in rows if r["section"] == "Stage 5 - sparsity"
               and r["label"] == "attention exceeds area null")
    assert row["n"] == 80
    assert row["point"] == 40 / 80


def test_sparsity_excludes_rows_where_attention_was_unavailable(tmp_path):
    """Recording those as failures would understate the metric with rows that
    were never measured -- the distinction the stage exists to preserve."""
    res = _sparsity_csv(tmp_path)
    rows = stage11.collect(res, "m", n_boot=200)
    row = next(r for r in rows if r["section"] == "Stage 5 - sparsity"
               and r["label"] == "attention exceeds area null")
    assert row["n"] == 80  # not 81


def test_sparsity_is_reported_per_rule(tmp_path):
    res = _sparsity_csv(tmp_path)
    rows = stage11.collect(res, "m", n_boot=200)
    by_rule = {r["label"]: r for r in rows
               if r["section"] == "Stage 5 - attention vs area null by rule"}
    assert by_rule["rule_1"]["point"] == 30 / 40
    assert by_rule["rule_4"]["point"] == 10 / 40


def _stability_row(image_id, baseline, verdicts, regions, error=""):
    from xai_vlm.metrics.stability import is_flickering, pairwise_agreement
    return {
        "image_id": image_id, "rule_id": "rule_1", "baseline_verdict": baseline,
        "verdicts": "|".join(verdicts), "regions": "|".join(regions),
        "agreement_rate": "" if error else f"{pairwise_agreement(verdicts):.4f}",
        "region_presence_rate": "", "flickering": "" if error else str(is_flickering(regions)),
        "mean_centroid_drift": "", "latency_ms": "1000", "error": error,
    }


def _stability_csv(tmp_path):
    """40 baseline-YES rows (30 unanimous) and 40 baseline-NO rows (10 unanimous).
    Regions flicker on 12 rows. Two errored rows must count nowhere."""
    box, rows = "(1.0, 1.0, 2.0, 2.0)", []
    for i in range(40):
        v = ["YES"] * 5 if i < 30 else ["YES", "NO", "YES", "YES", "YES"]
        r = [box] * 5 if i >= 12 else [box, "", box, box, box]
        rows.append(_stability_row(f"y-{i:03d}", "YES", v, r))
    for i in range(40):
        if i < 10:
            v = ["NO"] * 5
        elif i < 20:
            v = ["YES", "YES", "NO", "YES", "NO"]   # majority overturns NO
        else:
            v = ["NO", "YES", "NO", "YES", "NO"]
        rows.append(_stability_row(f"n-{i:03d}", "NO", v, [""] * 5))
    rows += [_stability_row(f"e-{i}", "YES", [], [], error="boom") for i in range(2)]
    _write(tmp_path / "stability_m.csv", rows)
    return tmp_path


def _by_label(rows, section):
    return {r["label"]: r for r in rows if r["section"] == section}


def test_stability_reports_unanimity_and_flicker(tmp_path):
    rows = stage11.collect(_stability_csv(tmp_path), "m", n_boot=200)
    got = _by_label(rows, "Stage 6 - stability")
    assert got["verdict unanimous (all reruns)"]["n"] == 80
    assert abs(got["verdict unanimous (all reruns)"]["point"] - 40 / 80) < 1e-9
    assert got["flickering region"]["n"] == 80
    assert abs(got["flickering region"]["point"] - 12 / 80) < 1e-9


def test_stability_is_split_by_baseline_verdict(tmp_path):
    """A model that is steady on YES and erratic on NO -- as this one is
    suspected to be -- would be averaged into one unremarkable number."""
    rows = stage11.collect(_stability_csv(tmp_path), "m", n_boot=200)
    got = _by_label(rows, "Stage 6 - unanimity by baseline verdict")
    assert abs(got["baseline YES"]["point"] - 30 / 40) < 1e-9
    assert abs(got["baseline NO"]["point"] - 10 / 40) < 1e-9


def test_stability_reports_majority_disagreeing_with_greedy_baseline(tmp_path):
    """Sampling at T=0.7 against the greedy Stage 2 verdict: how often does the
    majority of reruns overturn the answer the paper reports?"""
    rows = stage11.collect(_stability_csv(tmp_path), "m", n_boot=200)
    got = _by_label(rows, "Stage 6 - stability")
    # YES rows keep a YES majority; NO rows 10..19 get a YES majority.
    assert abs(got["majority overturns baseline"]["point"] - 10 / 80) < 1e-9
    assert got["majority overturns baseline"]["n"] == 80


def _baseline_row(image_id, stratum, expected, verdict, rule="rule_1"):
    return {"image_id": image_id, "rule_id": rule, "stratum": stratum,
            "is_negative_control": str(expected == "NO"), "expected": expected,
            "verdict": verdict, "region": "", "rationale": "", "confidence": "",
            "raw_text": "", "latency_ms": "1000", "parse_strict": "True",
            "parse_error": "", "error": ""}


def _whole_corpus(tmp_path):
    """Sample: 40 H (all YES) + 40 C (10 say NO). Complement: 120 R (90 say NO)."""
    sample = [_baseline_row(f"h{i}", "H", "YES", "YES") for i in range(40)]
    sample += [_baseline_row(f"c{i}", "C", "NO", "NO" if i < 10 else "YES") for i in range(40)]
    rest = [_baseline_row(f"r{i}", "R", "NO", "NO" if i < 90 else "YES",
                          rule="rule_2" if i % 2 else "rule_1") for i in range(120)]
    _write(tmp_path / "baseline_m.csv", sample)
    _write(tmp_path / "baseline_m_complement.csv", rest)
    return tmp_path


def test_whole_corpus_specificity_pools_matched_and_unsampled_compliant(tmp_path):
    got = _by_label(stage11.collect(_whole_corpus(tmp_path), "m", n_boot=200),
                    "Stage 2 - every compliant image")
    assert abs(got["specificity, matched (C)"]["point"] - 10 / 40) < 1e-9
    assert abs(got["specificity, unsampled (R)"]["point"] - 90 / 120) < 1e-9
    assert got["specificity, all compliant"]["n"] == 160
    assert abs(got["specificity, all compliant"]["point"] - 100 / 160) < 1e-9


def test_whole_corpus_section_is_absent_without_the_complement_run(tmp_path):
    _whole_corpus(tmp_path)
    (tmp_path / "baseline_m_complement.csv").unlink()
    rows = stage11.collect(tmp_path, "m", n_boot=200)
    assert not [r for r in rows if r["section"].startswith("Stage 2 - every compliant")]


def test_whole_corpus_false_alarms_are_split_by_rule(tmp_path):
    got = _by_label(stage11.collect(_whole_corpus(tmp_path), "m", n_boot=200),
                    "Stage 2 - false-alarm rate by rule, all compliant")
    assert set(got) == {"rule_1", "rule_2"}
    assert got["rule_1"]["n"] + got["rule_2"]["n"] == 160


def _completeness_attention_csv(tmp_path):
    """40 no region, 30 weak (lift 0.5), 50 supported (20 at lift 3, 30 at 1.5)."""
    rows = [{"image_id": f"n{i}", "completeness": "no_usable_explanation",
             "attention_lift": ""} for i in range(40)]
    rows += [{"image_id": f"w{i}", "completeness": "explanation_weak",
              "attention_lift": "0.5"} for i in range(30)]
    rows += [{"image_id": f"s{i}", "completeness": "explanation_supported",
              "attention_lift": "3.0" if i < 20 else "1.5"} for i in range(50)]
    _write(tmp_path / "completeness_attention_m.csv", rows)
    return tmp_path


def test_attentional_completeness_is_reported_over_every_row(tmp_path):
    got = _by_label(stage11.collect(_completeness_attention_csv(tmp_path), "m", n_boot=200),
                    "Stage 9 - completeness (attentional, not causal)")
    assert got["explanation supported"]["n"] == 120
    assert abs(got["explanation supported"]["point"] - 50 / 120) < 1e-9
    assert abs(got["explanation weak"]["point"] - 30 / 120) < 1e-9
    assert abs(got["no usable explanation"]["point"] - 40 / 120) < 1e-9
    # Among rows that named a region: how many sit well clear of chance.
    assert got["lift >= 2x (regions only)"]["n"] == 80
    assert abs(got["lift >= 2x (regions only)"]["point"] - 20 / 80) < 1e-9


def test_the_ablation_completeness_is_never_reported(tmp_path):
    """F7: computed, kept, and not quotable."""
    _write(tmp_path / "completeness_m.csv",
           [{"image_id": "a", "completeness": "explanation_supported"}])
    rows = stage11.collect(tmp_path, "m", n_boot=200)
    assert not [r for r in rows if r["section"].startswith("Stage 9")]
