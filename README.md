# Explainable AI for Construction Safety VLMs

Research project (advised by Professor Mustafa Abdallah) adapting his six-metric XAI
evaluation framework — originally built for tabular network-intrusion detection models —
to Vision-Language Models (VLMs) used for construction-site safety monitoring.

The core question: do VLM explanations remain accurate, focused, repeatable, efficient,
robust to noise, and necessary (not just sufficient) once you move from tabular features to
image + text inputs? A two-week technical pilot (Florence-2 on the ConstructionSite 10k
dataset) was run to test that before committing to a full study.

## Repository layout

| Folder | Contents |
|---|---|
| [`proposal/`](proposal/) | Research proposal (Markdown/LaTeX/Word sources and rendered PDF/DOCX). |
| [`planning/`](planning/) | Early methodology translation notes, dataset-mapping findings, the two-week pilot plan. |
| [`pilot/`](pilot/) | The technical pilot: Python pipeline, tests, results, and the LaTeX pilot report. See [`pilot/README.md`](pilot/README.md). |
| [`papers/`](papers/) | Reference papers (Professor Abdallah's XAI framework paper, a Whitebox XAI survey). |
| [`docs/`](docs/) | Planning specs/plans written while building the pilot report. |
| `feedbacks-for-full-project.md`, `my-notes.md` | Running notes and feedback log. |

## Status

The technical pilot (`pilot/`) is complete: all 11 pipeline scripts and 6 metrics ran
end-to-end on a 163-sample draw from ConstructionSite 10k, plus a from-scratch
reproducibility rerun. Findings are written up day-by-day in
[`pilot/report/chapters/`](pilot/report/chapters/) and summarized in
[`pilot/report/pilot_report_draft.md`](pilot/report/pilot_report_draft.md).

## Setup

Each subproject keeps its own dependencies; see [`pilot/README.md`](pilot/README.md) for
the pilot's Python environment and how to run the pipeline.
