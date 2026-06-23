# Design: Day-by-Day LaTeX Chapters for the XAI Pilot

## Purpose

Turn the 14-day Florence-2 / ConstructionSite-10k XAI pilot (`pilot/`) into 14
academic-paper-register LaTeX chapters, one per pilot day, compiled into a
single combined PDF report. Each chapter is written so it can be lifted
near-verbatim into sections of an eventual research paper with Professor
Mustafa Abdallah, not as an internal diary/status update.

Source of truth for every claim: existing `pilot/results/dayN_findings.md`
files, the actual source code in `pilot/src/xai_pilot/`, `pilot/scripts/`,
result CSVs in `pilot/results/`, figures in `pilot/results/figures/`, git
commit messages, and `pilot/report/*.md`. Nothing is invented. Where source
material for a given day is thin, the chapter stays shorter rather than
padding with invented narrative or numbers — this mirrors the rigor already
established in `pilot/report/pilot_report_draft.md` ("every number traces to
a specific CSV or figure; none are invented or estimated").

## Decisions (from brainstorming Q&A)

1. **Register**: formal academic paper draft chapters, not diary/status
   narrative.
2. **Depth**: scales to available evidence per day — rich `findings.md` days
   (3, 5, 6, 7, 8, 9, 10, 11, 13) get full treatment (~4-6 pages); thin days
   (1, 2, 4, 12, 14) stay leaner (~2-3 pages).
3. **Architecture diagrams**: TikZ, only at real turning points (6 total),
   not one per day. Other chapters note "no architecture change" in prose
   instead of a near-duplicate diagram.
4. **Location/style**: `pilot/report/chapters/`, reusing the color/heading/
   table style already defined in `proposal/Research-prposal-latex.tex`.
5. **Chapter titles**: methodological, with the day number as a parenthetical
   subtitle (e.g. "Chapter 3: From Scene-Level to Person-Relative Grounding —
   Redesigning the Safety-Rule Proxy (Day 3)").
6. **Bibliography**: a real `references.bib`, citations verified (not
   trusted from memory alone) before finalizing.
7. **Per-chapter structure**: flexible common spine, not a rigid skeleton —
   sections appear only when that day actually produced that kind of content.

## File layout

```
pilot/report/chapters/
  main.tex              -- report-class document; \include's all 14 chapters + bib
  preamble.tex           -- shared style lifted from proposal/Research-prposal-latex.tex
  references.bib
  day01_environment.tex
  day02_sample_selection.tex
  day03_baseline_inference.tex
  day04_explanation_regions.tex
  day05_descriptive_accuracy.tex
  day06_visual_sparsity.tex
  day07_stability.tex
  day08_efficiency.tex
  day09_robustness.tex
  day10_bounded_completeness.tex
  day11_analysis_figures.tex
  day12_pilot_report.tex
  day13_reproducibility.tex
  day14_meeting_package.tex
```

Figures referenced via relative `\includegraphics` paths into
`../../results/figures/...` (no copying/duplication of the real PNGs).

Build: `latexmk -pdf main.tex` (MiKTeX confirmed installed locally:
pdflatex/latexmk/xelatex all present). Compile incrementally — after every
2-3 chapters, not only once at the end — to catch LaTeX errors early.

## Chapter list (titles + primary sources + architecture diagram)

| # | Title | Day | Primary sources | Arch. diagram? |
|---|---|---|---|---|
| 1 | Establishing a Reproducible GPU Environment for Florence-2 | 1 | `scripts/01_check_environment.py`, commit `c30c4a9`, verified rerun output (real GPU run captured during the repo audit: `cuda:0`, fp16, caption "A large truck with a crane on the back of it.", 1565.3 ms) | Yes — initial pipeline (Diagram A) |
| 2 | Constructing a Balanced, Rule-Mapped Pilot Sample | 2 | `data.py`, `scripts/02_select_samples.py`, commit `c76a642`, `data/pilot_samples.csv` value counts (50/50/13/50) | No |
| 3 | Baseline Inference and the Discovery of a Sensitivity Failure | 3 | `day3_findings.md`, `inference.py`, commit `d4c0cca` (3.0%→27.0% sensitivity, 9x) | Yes — proxy v1 (scene-level) → v2 (person-relative) (Diagram B) |
| 4 | Standardizing Explanation Regions for Masking-Based Tests | 4 | `regions.py`, commit `6de76fb` (159/163 real box, 4/163 grid fallback) | Yes — regions/masking layer added (Diagram C) |
| 5 | Descriptive Accuracy: Does Masking the Top Region Change the Answer? | 5 | `day5_findings.md`, `descriptive_accuracy.py`, `results/descriptive_accuracy.csv` | No |
| 6 | Visual Sparsity via Decoder→Encoder Cross-Attention | 6 | `day6_findings.md`, `attribution.py`, `sparsity.py` | Yes — attribution layer added, rollout-vs-cross-attention rationale (Diagram D) |
| 7 | Stability Under Stochastic Decoding | 7 | `day7_findings.md`, `stability.py` | No |
| 8 | Efficiency and the Cost of Six Metrics at Scale | 8 | `day8_findings.md`, `efficiency.py` | No |
| 9 | Robustness to Perturbation and Prompt Rewording | 9 | `day9_findings.md`, `perturbations.py`, `robustness.py` | Yes — perturbation layer added (Diagram E) |
| 10 | Bounded Completeness: Is the Top Region Actually Necessary? | 10 | `day10_findings.md`, `completeness.py` | No |
| 11 | Aggregating Six Metrics into a Single Cross-Class Picture | 11 | `day11_findings.md`, `11_make_figures.py`, `pilot_metric_summary.csv` | Yes — final full-pipeline diagram (Diagram F) |
| 12 | Synthesizing Findings into a Pilot Report | 12 | `pilot_report_draft.md`, commit `c2a1f85` | No |
| 13 | From-Scratch Reproducibility — and a Real Bug Found by Testing the Process | 13 | `day13_findings.md` (hardcoded `n=50` chart-label bug, found & fixed) | No |
| 14 | Packaging the Pilot for a Research Collaboration Meeting | 14 | `one_pager_summary.md`, `talking_points.md`, `full_study_task_list.md` | No |

## Per-chapter skeleton (flexible)

Common spine, sections included only when the day produced that kind of
content:

Motivation → Method/Implementation → Architecture Before/After (turning
points only) → Worked Example(s) with real figures from `results/figures/`
→ Results → Problems Encountered & Resolutions → Standard vs. Non-Standard
Choices → Limitations → Implications for the Paper.

## Bibliography

`references.bib`, verified against the actual PDFs in `papers/` and via web
search before finalizing (not trusted from memory alone):

- Arreche, Guntur, Roberts & Abdallah, "E-XAI: Evaluating Black-Box
  Explainable AI Frameworks for Network Intrusion Detection," IEEE Access,
  vol. 12, pp. 23954–, 2024. (Source of the six-metric framework.)
- Arreche & Abdallah, "A comparative analysis of DNN-based white-box
  explainable AI methods in network security," EURASIP Journal on
  Information Security, 2025:16.
- Florence-2 (Xiao et al., CVPR 2024).
- Attention Rollout (Abnar & Zuidema, ACL 2020) — cited as the method
  Day 6 deviated from, with the reasoning why.
- Captum (Kokhlikyan et al., 2020), LRP (Bach et al., 2015), DeepLIFT
  (Shrikumar et al., 2017) — cited as attribution-method alternatives
  discussed but not used.
- `LouisChen15/ConstructionSite` ("ConstructionSite 10k") — **no known
  associated paper; will not be fabricated.** Cited as a dataset resource
  (HF Hub URL) only, `@misc` entry.

## Verification plan

- Compile `main.tex` with `latexmk -pdf` after every 2-3 chapters; fix
  LaTeX errors immediately rather than batching them.
- Every numeric claim in a chapter must trace to a specific file
  (`dayN_findings.md`, a `results/*.csv`, or source code) — same rule the
  existing `pilot_report_draft.md` already follows.
- Final pass: confirm the combined PDF builds clean end-to-end with no
  missing figures/citations (`\includegraphics` paths resolve, no
  `??` citation markers).

## Out of scope for this pass

- Editing the existing `pilot_report_draft.md` or `report/*.md` files —
  this is a new, separate set of chapters, not a replacement.
- Submitting/sending the compiled PDF anywhere — local artifact only.
- Re-running any GPU pipeline scripts to generate new figures — chapters
  use the figures that already exist in `pilot/results/figures/`.
