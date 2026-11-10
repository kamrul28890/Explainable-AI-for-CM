"""Regenerate every headline figure, with confidence intervals, from committed CSVs.

ONE COMMAND, ONE SOURCE OF TRUTH. Numbers quoted in the paper come from here, not
from a summary typed by hand into a README, so a figure in the text can always be
traced to the file and the sample it came from.

Every rate carries a bootstrap interval and its n. A paper that prints "59%"
without an interval invites the reader to treat it as exact, and hides the
difference between a rate over 2,555 rows and one over 24 -- which matters here,
because rule_2 and rule_4 have an order of magnitude fewer positives than rule_1.

    python scripts/11_report.py --model qwen7b-4bit
    python scripts/11_report.py --model qwen7b-4bit --format latex
"""
import argparse
import csv
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402

from xai_vlm.metrics.reporting import MIN_N_FLOOR, rate_row  # noqa: E402


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _truthy(v) -> bool:
    return str(v).strip().lower() == "true"


def collect(res: Path, model: str, n_boot: int) -> list[dict]:
    """Every reportable rate, in the order the paper presents them."""
    rows: list[dict] = []

    baseline = _read(res / f"baseline_{model}.csv")
    if baseline:
        pos = [r for r in baseline if r["expected"] == "YES"]
        neg = [r for r in baseline if r["expected"] == "NO"]
        controls = [r for r in baseline if _truthy(r.get("is_negative_control"))]

        rows.append({"section": "Stage 2 - baseline", **rate_row(
            "sensitivity", [r["verdict"] == "YES" for r in pos], n_boot)})
        rows.append({"section": "Stage 2 - baseline", **rate_row(
            "specificity", [r["verdict"] == "NO" for r in neg], n_boot)})
        rows.append({"section": "Stage 2 - baseline", **rate_row(
            "false-alarm rate", [r["verdict"] == "YES" for r in controls], n_boot)})
        rows.append({"section": "Stage 2 - baseline", **rate_row(
            "YES share", [r["verdict"] == "YES" for r in baseline], n_boot)})
        rows.append({"section": "Stage 2 - baseline", **rate_row(
            "parse rate", [bool(r["verdict"]) for r in baseline], n_boot)})

        # Per-rule sensitivity. rule_2 and rule_4 are the reason the floor exists.
        for rule in sorted({r["rule_id"] for r in baseline}):
            sub = [r for r in pos if r["rule_id"] == rule]
            if sub:
                rows.append({"section": "Stage 2 - sensitivity by rule", **rate_row(
                    rule, [r["verdict"] == "YES" for r in sub], n_boot)})

    # The complement run: every compliant image the sample left out, asked under
    # the same policy as stratum C. Pooled with C it is a whole-corpus specificity
    # that needs no prevalence reweighting. C and R are also shown apart, because
    # C was matched to the hazard stratum's scene conditions (it holds 83 of the
    # 87 night images) and R was not.
    rest = _read(res / f"baseline_{model}_complement.csv")
    if baseline and rest:
        matched = [r for r in baseline if r.get("stratum") == "C"]
        everything = matched + rest
        sec = "Stage 2 - every compliant image"
        for label, rows_ in (("specificity, matched (C)", matched),
                             ("specificity, unsampled (R)", rest),
                             ("specificity, all compliant", everything)):
            rows.append({"section": sec, **rate_row(
                label, [r["verdict"] == "NO" for r in rows_], n_boot)})
        for rule in sorted({r["rule_id"] for r in everything}):
            sub = [r for r in everything if r["rule_id"] == rule]
            rows.append({"section": "Stage 2 - false-alarm rate by rule, all compliant",
                         **rate_row(rule, [r["verdict"] == "YES" for r in sub], n_boot)})

    regions = _read(res / f"regions_{model}.csv")
    if regions:
        rows.append({"section": "Stage 3 - regions", **rate_row(
            "no-region rate (all)", [r["source"] == "none" for r in regions], n_boot)})
        yes = [r for r in regions if r["verdict"] == "YES"]
        rows.append({"section": "Stage 3 - regions", **rate_row(
            "no-region rate (YES only)", [r["source"] == "none" for r in yes], n_boot)})

    corr = _read(res / f"explanation_correctness_{model}.csv")
    if corr:
        scored = [r for r in corr if not r["undefined_reason"]]
        rows.append({"section": "Stage 10 - explanation correctness", **rate_row(
            "explanation correct", [r["verdict_correctness"] == "CORRECT" for r in scored],
            n_boot)})
        rows.append({"section": "Stage 10 - explanation correctness", **rate_row(
            "similarity >= 0.60", [float(r["similarity"]) >= 0.60 for r in scored], n_boot)})
        rows.append({"section": "Stage 10 - explanation correctness", **rate_row(
            "hazard match", [_truthy(r["hazard_ok"]) for r in scored], n_boot)})
        rows.append({"section": "Stage 10 - explanation correctness", **rate_row(
            "IoU >= 0.30", [float(r["iou"]) >= 0.30 for r in scored], n_boot)})

        right = [r for r in scored if r["verdict"] == "YES" and r["expected"] == "YES"]
        if right:
            rows.append({"section": "Stage 10 - explanation correctness", **rate_row(
                "right answer, wrong reason",
                [r["verdict_correctness"] == "INCORRECT" for r in right], n_boot)})

        for q in sorted({r["quality_of_info"] for r in scored if r["quality_of_info"]}):
            sub = [r for r in scored if r["quality_of_info"] == q]
            rows.append({"section": "Stage 10 - by quality_of_info", **rate_row(
                q, [r["verdict_correctness"] == "CORRECT" for r in sub], n_boot)})

    da = _read(res / f"descriptive_accuracy_{model}.csv")
    if da:
        yes = [r for r in da if r["baseline_verdict"] == "YES"]
        no = [r for r in da if r["baseline_verdict"] == "NO"]

        # One section per table, both methods inside it. Interleaving the
        # sections would make render_latex open a fresh tabular per method and
        # emit four half-tables where the paper needs two.
        for method in ("black", "tokens"):
            if yes:
                rows.append({"section": "Stage 4 - descriptive accuracy", **rate_row(
                    f"flip rate ({method})",
                    [_truthy(r[f"flip_{method}"]) for r in yes], n_boot)})

        # Per rule, and YES rows only. The controls are rule-labelled too, so
        # pooling them in would fold the artefact floor into the signal that is
        # supposed to be measured against it.
        for method in ("black", "tokens"):
            for rule in sorted({r["rule_id"] for r in yes}):
                sub = [r for r in yes if r["rule_id"] == rule]
                rows.append({"section": f"Stage 4 - flip rate by rule ({method})",
                             **rate_row(
                                 rule, [_truthy(r[f"flip_{method}"]) for r in sub],
                                 n_boot)})

        for method in ("black", "tokens"):
            if no:
                rows.append({"section": "Stage 4 - artefact floor", **rate_row(
                    f"NO->YES ({method})",
                    [_truthy(r[f"flip_{method}"]) for r in no], n_boot)})
        # The two maskings answer the same question about the same region.
        # Where they disagree, at least one of them is reporting the mask rather
        # than the model, so this belongs beside the floor.
        rows.append({"section": "Stage 4 - artefact floor", **rate_row(
            "methods disagree",
            [_truthy(r["methods_disagree"]) for r in da], n_boot)})

    sp = _read(res / f"sparsity_{model}.csv")
    if sp:
        # Raw attention mass is not interpretable on its own: a region covering
        # half the image should attract half the attention by chance alone. The
        # reportable quantity is mass against THAT ROW'S OWN area fraction, which
        # is what uniform attention would put there.
        #
        # Rows where attention was unavailable are excluded rather than counted
        # as misses -- "not measured" is not "measured and failed", which is the
        # distinction Stage 5 was written to preserve (and M5).
        scored = [r for r in sp
                  if r.get("attention_available") == "True"
                  and r.get("mass_inside_region")
                  and r.get("region_area_fraction")]
        if scored:
            beats = [float(r["mass_inside_region"]) > float(r["region_area_fraction"])
                     for r in scored]
            rows.append({"section": "Stage 5 - sparsity", **rate_row(
                "attention exceeds area null", beats, n_boot)})

            for rule in sorted({r["rule_id"] for r in scored}):
                sub = [r for r in scored if r["rule_id"] == rule]
                rows.append({"section": "Stage 5 - attention vs area null by rule",
                             **rate_row(
                                 rule,
                                 [float(r["mass_inside_region"])
                                  > float(r["region_area_fraction"]) for r in sub],
                                 n_boot)})

    st = _read(res / f"stability_{model}.csv")
    if st:
        # Rows that errored have no agreement rate and count nowhere: "not
        # measured" again, not "measured and unstable".
        ok = [r for r in st if r.get("agreement_rate") and not r.get("error")]
        if ok:
            # Unanimity, not the mean pairwise agreement: a median of 100% hid a
            # quarter of rows at 60% or worse in the first run, and "all five
            # reruns gave the same answer" is the property a user relies on.
            unanimous = lambda r: float(r["agreement_rate"]) == 1.0  # noqa: E731
            rows.append({"section": "Stage 6 - stability", **rate_row(
                "verdict unanimous (all reruns)", [unanimous(r) for r in ok], n_boot)})
            rows.append({"section": "Stage 6 - stability", **rate_row(
                "flickering region", [_truthy(r["flickering"]) for r in ok], n_boot)})

            # Reruns sample at T=0.7; Stage 2 is greedy. A strict majority of
            # reruns against the greedy verdict means the reported answer is
            # the minority one under the model's own distribution.
            def overturns(r) -> bool:
                vs = [v for v in r["verdicts"].split("|") if v]
                other = sum(1 for v in vs if v != r["baseline_verdict"])
                return other * 2 > len(vs)
            rows.append({"section": "Stage 6 - stability", **rate_row(
                "majority overturns baseline", [overturns(r) for r in ok], n_boot)})

            for v in ("YES", "NO"):
                sub = [r for r in ok if r["baseline_verdict"] == v]
                if sub:
                    rows.append({"section": "Stage 6 - unanimity by baseline verdict",
                                 **rate_row(f"baseline {v}", [unanimous(r) for r in sub],
                                            n_boot)})

    return rows


def render_text(rows: list[dict]) -> str:
    out, section = [], None
    for r in rows:
        if r["section"] != section:
            section = r["section"]
            out.append("")
            out.append(section)
            out.append("-" * len(section))
        flag = "  <-- " + r["note"] if r["below_floor"] else ""
        out.append(f"  {r['label']:<32} {r['formatted']}{flag}")
    return "\n".join(out)


def render_latex(rows: list[dict]) -> str:
    """A booktabs table per section, ready to \\input into the paper."""
    out, section = [], None
    for r in rows:
        if r["section"] != section:
            if section is not None:
                out += [r"\bottomrule", r"\end{tabular}", ""]
            section = r["section"]
            out += [f"% {section}",
                    r"\begin{tabular}{lrrr}", r"\toprule",
                    r"Metric & Estimate & 95\% CI & $n$ \\", r"\midrule"]
        ci = f"[{r['lo'] * 100:.1f}, {r['hi'] * 100:.1f}]"
        dagger = r"$^{\dagger}$" if r["below_floor"] else ""
        label = r["label"].replace("_", r"\_").replace(">=", r"$\geq$")
        out.append(f"{label}{dagger} & {r['point'] * 100:.1f}\\% & {ci} & {r['n']} \\\\")
    if section is not None:
        out += [r"\bottomrule", r"\end{tabular}"]
    out.append("")
    out.append(rf"% $\dagger$ below the reporting floor of n={MIN_N_FLOOR}; "
               "illustrative only")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen7b-4bit")
    ap.add_argument("--format", choices=("text", "latex"), default="text")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = config.RESULTS_DIR
    rows = collect(res, args.model, args.n_boot)
    if not rows:
        print("no result files found; run the stages first.", file=sys.stderr)
        return 1

    body = render_latex(rows) if args.format == "latex" else render_text(rows)
    header = (f"Study 2 headline figures -- {args.model}\n"
              f"bootstrap 95% CI, {args.n_boot} resamples, "
              f"reporting floor n={MIN_N_FLOOR}")
    text = header + "\n" + body if args.format == "text" else body

    if args.out:
        # Bytes, not write_text: Windows text mode turns \n into \r\n, so the same
        # CSVs would give a different report file on each machine.
        Path(args.out).write_bytes((text + "\n").encode("utf-8"))
        print(f"wrote {args.out}")
    else:
        print(text)

    below = [r for r in rows if r["below_floor"]]
    if below and args.format == "text":
        print(f"\n{len(below)} rate(s) below the n={MIN_N_FLOOR} reporting floor "
              "and marked as illustrative.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
