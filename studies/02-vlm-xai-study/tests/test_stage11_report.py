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
