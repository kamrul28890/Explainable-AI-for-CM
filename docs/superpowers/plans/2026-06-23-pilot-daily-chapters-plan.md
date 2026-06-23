# Pilot Daily Chapters (LaTeX) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task (sequential, inline execution — preserves narrative voice across chapters; do NOT use subagent-driven-development for this plan, see rationale in the handoff section at the end). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 14-day Florence-2/ConstructionSite-10k XAI pilot into 14 academic-paper-register LaTeX chapters (`pilot/report/chapters/day01_*.tex` ... `day14_*.tex`), compiled into one combined PDF (`pilot/report/chapters/main.tex`).

**Architecture:** A `report`-class LaTeX document with a shared `preamble.tex` (styled after `proposal/Research-prposal-latex.tex`), one `\chapter`-per-day `.tex` file `\include`d from `main.tex`, a verified `references.bib`, and 6 TikZ architecture diagrams at real pipeline turning points (Chapters 1, 3, 4, 6, 9, 11).

**Tech Stack:** LaTeX (MiKTeX, confirmed installed: pdflatex/latexmk/xelatex), `report` document class, `natbib`, `tikz`, `subcaption`, `booktabs`/`longtable`. Build command: `latexmk -pdf main.tex` run from `pilot/report/chapters/`.

## Adaptation note for this plan (read first)

This is a **content-authoring plan, not a code plan**. The writing-plans template assumes TDD/code; here, "the test" for each task is **does it compile clean (no LaTeX errors, no missing `\includegraphics` files, no undefined `\cite` keys)**, and "no placeholders" means **every fact, number, and figure path below is real and pre-verified** (pulled directly from `pilot/results/dayN_findings.md`, source code, or result CSVs already read during planning) — not that this document contains finished prose. Writing the connecting sentences between these pre-verified facts is the actual work of each task.

## Global Constraints

- Every numeric claim in a chapter must trace to a specific source file (a `dayN_findings.md`, a `results/*.csv`, or source code) — same evidentiary rule `pilot/report/pilot_report_draft.md` already follows. Do not invent or round numbers beyond what's given below.
- Figures: reference the **real, already-existing** PNGs in `pilot/results/figures/` via `\includegraphics`. Never create new figures or re-run GPU scripts.
- `ConstructionSite 10k` has **no known associated academic paper** — cite it only as the `@misc` dataset entry (`constructionsite10k`), never invent an author/venue for it.
- Compile after every single task (not batched) — each task's deliverable is independently testable by compiling `main.tex`.
- Commit after every task (local commits, per user's "commit as I go" instruction from brainstorming).
- Chapter titles are methodological, with the day noted parenthetically (e.g. "Chapter 3: ... (Day 3)") — decided in brainstorming, do not revert to plain "Day N" titles.

---

## Task 1: Scaffold the LaTeX project (preamble, empty main.tex, directory)

**Files:**
- Create: `pilot/report/chapters/preamble.tex`
- Create: `pilot/report/chapters/main.tex`
- Create: `pilot/report/chapters/references.bib` (empty stub, populated in Task 2)

**Interfaces:**
- Produces: `\graphicspath{{../../results/figures/}}` (so every later chapter can `\includegraphics{subdir/file.png}` without repeating the relative path); TikZ node styles `block` and `arrow` (used by Diagrams A-F in Tasks 3, 5, 6, 8, 11, 13); color names `inkblue`, `headingblue`, `darkblue`, `mutedgray`, `tablefill`, `rulegray` (reused from `proposal/Research-prposal-latex.tex` for visual consistency).

- [ ] **Step 1: Write `preamble.tex`**

```latex
\usepackage[margin=0.9in,headheight=22pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{array}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,shapes.geometric,fit}
\usepackage[numbers,sort&compress]{natbib}

\graphicspath{{../../results/figures/}}

\definecolor{inkblue}{HTML}{0B2545}
\definecolor{headingblue}{HTML}{2E74B5}
\definecolor{darkblue}{HTML}{1F4D78}
\definecolor{mutedgray}{HTML}{555555}
\definecolor{tablefill}{HTML}{F4F6F9}
\definecolor{rulegray}{HTML}{D0D7DE}

\setlength{\parindent}{0pt}
\setlength{\parskip}{7pt}
\renewcommand{\arraystretch}{1.25}
\setlist[itemize]{leftmargin=1.2em,itemsep=2pt,topsep=2pt}
\setlist[enumerate]{leftmargin=1.5em,itemsep=2pt,topsep=2pt}
\emergencystretch=2em

\titleformat{\chapter}[display]{\bfseries\color{inkblue}}{\large\color{mutedgray}\chaptertitlename\ \thechapter}{4pt}{\Large}
\titleformat{\section}{\large\bfseries\color{headingblue}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\normalsize\bfseries\color{headingblue}}{\thesubsection}{0.7em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{darkblue}}{\thesubsubsection}{0.7em}{}
\titlespacing*{\section}{0pt}{14pt}{5pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}
\titlespacing*{\subsubsection}{0pt}{8pt}{3pt}

\pagestyle{fancy}
\fancyhf{}
\lhead{\footnotesize\color{mutedgray}Pilot Chapters}
\rhead{\footnotesize\color{mutedgray}Explainable VLMs for Construction Safety}
\cfoot{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{rulegray}\leaders\hrule height \headrulewidth\hfill}}

\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcommand{\tableheader}{\rowcolor{tablefill}}

\tikzset{
  block/.style={rectangle, draw=darkblue, fill=tablefill, rounded corners, minimum height=1cm, minimum width=2.6cm, align=center, font=\small, text=inkblue},
  arrow/.style={-{Triangle[length=2.2mm,width=2mm]}, thick, darkblue},
  diagcaption/.style={font=\footnotesize\color{mutedgray}}
}
```

- [ ] **Step 2: Write `main.tex` with no chapters included yet**

```latex
\documentclass[11pt,letterpaper]{report}
\input{preamble}

\title{\bfseries\color{inkblue} Adapting a Six-Metric XAI Evaluation Framework to a Construction-Safety Vision-Language Model: A Fourteen-Day Pilot}
\author{Prepared for research collaboration with Professor Mustafa Abdallah}
\date{\today}

\begin{document}
\maketitle
\tableofcontents

% CHAPTERS — each later task appends one \include line here, in day order.

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
```

- [ ] **Step 3: Write an empty `references.bib` stub**

```bibtex
% Populated in Task 2. Keys used by later chapters:
% arreche2024exai, arreche2025whitebox, xiao2024florence2,
% abnar2020rollout, kokhlikyan2020captum, bach2015lrp,
% shrikumar2017deeplift, constructionsite10k
```

- [ ] **Step 4: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0, `main.pdf` created with a title page and an empty (no-entries) table of contents — no `! LaTeX Error` lines in `main.log`.

- [ ] **Step 5: Commit**

```bash
git add pilot/report/chapters/preamble.tex pilot/report/chapters/main.tex pilot/report/chapters/references.bib
git commit -m "Scaffold LaTeX chapter report (preamble, main.tex, empty bib)"
```

---

## Task 2: Build and verify the bibliography

**Files:**
- Modify: `pilot/report/chapters/references.bib`
- Modify: `pilot/report/chapters/main.tex` (temporary verification only, reverted in Step 3)

**Interfaces:**
- Produces: 8 verified `.bib` keys for all later chapters to `\citep{}`/`\citet{}`: `arreche2024exai`, `arreche2025whitebox`, `xiao2024florence2`, `abnar2020rollout`, `kokhlikyan2020captum`, `bach2015lrp`, `shrikumar2017deeplift`, `constructionsite10k`.

- [ ] **Step 1: Write the full `references.bib`**

All entries below were verified (PDF first-page reads for the two Abdallah papers; web search for the other five) during planning — do not alter author lists, venues, or identifiers.

```bibtex
@article{arreche2024exai,
  author  = {Arreche, Osvaldo and Guntur, Tanish R. and Roberts, Jack W. and Abdallah, Mustafa},
  title   = {E-XAI: Evaluating Black-Box Explainable AI Frameworks for Network Intrusion Detection},
  journal = {IEEE Access},
  volume  = {12},
  pages   = {23954--23988},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.3365140}
}

@article{arreche2025whitebox,
  author  = {Arreche, Osvaldo and Abdallah, Mustafa},
  title   = {A comparative analysis of {DNN}-based white-box explainable {AI} methods in network security},
  journal = {EURASIP Journal on Information Security},
  volume  = {2025},
  number  = {1},
  pages   = {16},
  year    = {2025},
  doi     = {10.1186/s13635-025-00201-x}
}

@inproceedings{xiao2024florence2,
  author       = {Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu, Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
  title        = {Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  booktitle    = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year         = {2024},
  eprint       = {2311.06242},
  archivePrefix= {arXiv}
}

@inproceedings{abnar2020rollout,
  author    = {Abnar, Samira and Zuidema, Willem},
  title     = {Quantifying Attention Flow in Transformers},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages     = {4190--4197},
  year      = {2020},
  doi       = {10.18653/v1/2020.acl-main.385}
}

@article{kokhlikyan2020captum,
  author  = {Kokhlikyan, Narine and Miglani, Vivek and Martin, Miguel and Wang, Edward and Alsallakh, Bilal and Reynolds, Jonathan and Melnikov, Alexander and Kliushkina, Natalia and Araya, Carlos and Yan, Siqi and Reblitz-Richardson, Orion},
  title   = {Captum: A unified and generic model interpretability library for {PyTorch}},
  journal = {arXiv preprint arXiv:2009.07896},
  year    = {2020}
}

@article{bach2015lrp,
  author  = {Bach, Sebastian and Binder, Alexander and Montavon, Gr{\'e}goire and Klauschen, Frederick and M{\"u}ller, Klaus-Robert and Samek, Wojciech},
  title   = {On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation},
  journal = {PLOS ONE},
  volume  = {10},
  number  = {7},
  pages   = {e0130140},
  year    = {2015},
  doi     = {10.1371/journal.pone.0130140}
}

@inproceedings{shrikumar2017deeplift,
  author    = {Shrikumar, Avanti and Greenside, Peyton and Kundaje, Anshul},
  title     = {Learning Important Features Through Propagating Activation Differences},
  booktitle = {Proceedings of the 34th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {70},
  pages     = {3145--3153},
  year      = {2017}
}

@misc{constructionsite10k,
  author       = {{LouisChen15}},
  title        = {{ConstructionSite}: A Construction-Site Safety Vision-Language Dataset},
  howpublished = {Hugging Face Hub dataset, \url{https://huggingface.co/datasets/LouisChen15/ConstructionSite}},
  year         = {2024},
  note         = {Accessed June 2026. No associated peer-reviewed publication was found at time of writing.}
}
```

- [ ] **Step 2: Temporarily verify all 8 entries render**

Edit `main.tex`, inserting directly above `\bibliographystyle{plainnat}`:
```latex
\nocite{*}
```
Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; `main.pdf`'s final pages list all 8 references, numbered `[1]`-`[8]`, plainnat-formatted; `main.log` contains no `Warning--I didn't find a database entry` lines.

- [ ] **Step 3: Remove the temporary `\nocite{*}` line**

Remove the line added in Step 2 — the bibliography should only list works actually `\cite`d by the finished chapters, not a dumped list. Recompile once to confirm `main.tex` still builds (the bibliography section will legitimately render empty/absent until Task 3 adds the first `\citep{xiao2024florence2}` — this is expected, not an error).

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/references.bib pilot/report/chapters/main.tex
git commit -m "Add verified bibliography (Abdallah's two XAI papers, Florence-2, attention rollout, Captum, LRP, DeepLIFT, dataset)"
```

---

## Task 3: Chapter 1 — Establishing a Reproducible GPU Environment for Florence-2 (Day 1)

**Files:**
- Create: `pilot/report/chapters/day01_environment.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day01_environment}` before `\bibliographystyle`)

**Interfaces:**
- Consumes: `\citep{xiao2024florence2}`, `\citep{constructionsite10k}` (from Task 2's bib).
- Produces: Diagram A (TikZ, "initial pipeline") — referenced again narratively (not redrawn) in Chapter 11's Diagram F discussion.

**Source material (all pre-verified, use exactly):**
- Goal per `two-week-pilot-plan.md` Day 1: confirm the pilot is executable — environment works, dataset sample loads, Florence-2 runs on one image.
- Commit `c30c4a9`: scaffolded `pilot/` as an installable package (`src/xai_pilot`) with `config.py` and `model.py`.
- `model.py` patches around Florence-2's `flash_attn` import (no usable Windows wheel) by monkeypatching `transformers.dynamic_module_utils.get_imports` to drop `flash_attn` from the import list, then forces `attn_implementation="sdpa"` in `from_pretrained`.
- `requirements.txt` pins `transformers==4.49.0`: `transformers>=5.0` removed legacy generation attributes (e.g. `forced_bos_token_id`) from `PretrainedConfig`, which crashes Florence-2's pinned remote-code `configuration_florence2.py` on load.
- `requirements.txt` also documents: `torch` must be installed first/separately from its pinned CUDA index (`torch==2.12.1+cu130`), because a plain `pip install torch` (or letting other deps pull it in) silently installs a CPU-only wheel.
- `config.py`: `HF_DATASET_ID = "LouisChen15/ConstructionSite"`, `MODEL_ID = "microsoft/Florence-2-base-ft"`, `SEED = 42`.
- `scripts/01_check_environment.py` loads the real ConstructionSite test split (streaming), loads Florence-2-base-ft, runs `<CAPTION>` on a real sample, exits non-zero if CUDA/dataset/model load fails.
- **Worked example — a real, freshly-reproduced run** (captured verbatim during report preparation, itself a small reproducibility check in the spirit of Day 13): `torch 2.12.1+cu130, cuda available: True`; `Sample image_id=0000001, size=(1200, 675)`; `Model device: cuda:0, dtype: torch.float16`; `Caption: A large truck with a crane on the back of it.`; `Mean token probability (confidence proxy): 0.6442`; `Inference time: 1565.3 ms`. State explicitly that this was reproduced, not merely quoted from Day 1's original log.
- Problems encountered & resolutions (report all three): (1) `flash_attn` import crash on Windows → monkeypatch `get_imports`; (2) `transformers>=5.0` config-load crash → version pin to 4.49.0; (3) silent CPU-only torch install → install the CUDA wheel first, separately, from the pinned index.
- Standard vs. non-standard: standard = `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`, a documented HF pattern for custom model code; non-standard = the `flash_attn` import patch, a Windows-specific workaround with no upstream documentation, and the hard version pin forced by an upstream breaking change.
- Florence-2 itself: 231.6M parameters, MIT license (from `pilot_report_draft.md` §3).

**Diagram A — TikZ code (initial pipeline, place in a `figure` environment with caption "Pipeline architecture after Day 1: environment and single-image inference only."):**

```latex
\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.6cm]
\node[block] (data) {ConstructionSite 10k\\(HF Hub, streaming)};
\node[block, right=of data] (model) {Florence-2-base-ft\\(CUDA, fp16, sdpa)};
\node[block, right=of model] (task) {\texttt{<CAPTION>}\\task token};
\node[block, right=of task] (out) {Caption +\\confidence proxy};
\draw[arrow] (data) -- (model);
\draw[arrow] (model) -- (task);
\draw[arrow] (task) -- (out);
\end{tikzpicture}
\caption{Pipeline architecture after Day 1: environment and single-image inference only.}
\label{fig:diagram-a}
\end{figure}
```

- [ ] **Step 1: Write `day01_environment.tex`**

Structure: `\chapter{Establishing a Reproducible GPU Environment for Florence-2 (Day 1)}`, then sections `Motivation`, `Implementation`, `Architecture` (Diagram A), `Worked Example` (the verified rerun output, formatted as a quoted block or table), `Problems Encountered and Resolutions` (the three fixes), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (recommend reporting software environment as a methods footnote/appendix, citing `xiao2024florence2` for the model and `constructionsite10k` for the dataset). Use the source material above for every factual claim — do not add numbers not listed here.

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day01_environment}` on its own line, immediately before `\bibliographystyle{plainnat}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; no `! LaTeX Error`, no `Undefined control sequence`, no `LaTeX Warning: Reference ... undefined` in `main.log`; Chapter 1 appears in the PDF's table of contents.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day01_environment.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 1: GPU environment setup (Day 1)"
```

---

## Task 4: Chapter 2 — Constructing a Balanced, Rule-Mapped Pilot Sample (Day 2)

**Files:**
- Create: `pilot/report/chapters/day02_sample_selection.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day02_sample_selection}`)

**Interfaces:**
- Consumes: nothing new from the bib (no external citation needed in this chapter).
- Produces: nothing new (no diagram; Chapter 3 references this chapter's class-imbalance fact directly in prose, not via a LaTeX label).

**Source material:**
- Commit `c76a642` (Day 2 half): `classify_image`/`assign_rule_id`/`select_balanced_sample` added to `data.py`; `RULE_QUERIES` added to `prompts.py`; `scripts/02_select_samples.py` writes `pilot_samples.csv`.
- `config.py`: `CLASS_PRIORITY = ["ppe_violation", "fall_hazard", "struck_by_risk", "compliant"]`; `RULE_TO_CLASS` maps `rule_1_violation→ppe_violation`, `rule_2_violation→fall_hazard`, `rule_3_violation→fall_hazard`, `rule_4_violation→struck_by_risk`; `SAMPLES_PER_CLASS = 50`; `SEED = 42`.
- `data.classify_image(row)`: a row violates 0+ rules; `primary_class` is the highest-priority violated class per `CLASS_PRIORITY` (a row violating both rule_1 and rule_3, e.g., is classified `ppe_violation`, not `fall_hazard`); `violated_rule_ids` lists every rule actually violated.
- `data.assign_rule_id`: violation classes get the one unambiguous violated rule that maps to `primary_class`; **compliant samples have no ground-truth rule**, so they are round-robined evenly across `COMPLIANT_ROUND_ROBIN = ["rule_1","rule_2","rule_3","rule_4"]` — a deliberate design choice to get balanced "model correctly says compliant" coverage per rule, rather than testing the same rule on every compliant image.
- `data.select_balanced_sample`: a single streaming pass over the dataset; a reservoir of candidates per class is kept, capped at `n_per_class * 10` (500) to bound memory; final selection is a seeded (`SEED=42`) random sample of `n_per_class` (50) from each class's reservoir.
- **Result**: 163 total rows, not the planned 200 — `ppe_violation 50, fall_hazard 50, compliant 50, struck_by_risk 13`. `struck_by_risk` capped at 13 by genuine dataset scarcity: the entire 3,004-image test split contains only 13 `struck_by_risk` examples, confirmed via a full-split scan, not a sampling bug.
- `prompts.py`'s `RULE_QUERIES`: two phrasings per rule — `rule_1: ["hard hat", "high-visibility vest"]`, `rule_2: ["safety harness", "fall-protection lanyard"]`, `rule_3: ["guardrail", "edge protection barrier"]`; `PERSON_PHRASE = "worker"`; `RULE_4_PROXIMITY_PAIR = ("worker", "excavator")`.
- Problems & resolutions: the `struck_by_risk` shortage was discovered by an exhaustive scan (not assumed), and the decision was made to proceed with n=13 for that class rather than block the pilot — explicitly flagged as a limitation that recurs in every later chapter's `struck_by_risk` numbers.
- Standard vs. non-standard: standard = stratified/balanced sampling via reservoir sampling with a fixed seed for reproducibility; non-standard/notable = the round-robin rule assignment specifically for compliant samples, a domain-specific balancing choice not found in generic sampling libraries.
- Implications for the paper: the class-imbalance finding (13 vs. 50) is a **dataset** limitation, not a method limitation, and should be reported in any paper's Dataset/Limitations section; it directly motivates the "scale vs. fix imbalance" open question raised again in Chapter 14.

- [ ] **Step 1: Write `day02_sample_selection.tex`**

Structure: `\chapter{Constructing a Balanced, Rule-Mapped Pilot Sample (Day 2)}`, sections `Motivation`, `Implementation` (the three functions + round-robin design choice), `Results` (163, not 200; the value-count table), `Problems Encountered and Resolutions`, `Standard vs. Non-Standard Choices`, `Implications for the Paper`. No figures, no diagram, no new citations — keep this chapter lean (~2-3 pages) per the design's "scale to evidence" rule.

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day02_sample_selection}` immediately after `\include{day01_environment}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; both Chapter 1 and Chapter 2 appear in the table of contents; no new errors/warnings in `main.log` beyond what Task 3 already produced.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day02_sample_selection.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 2: balanced sample selection (Day 2)"
```

---

## Task 5: Chapter 3 — Baseline Inference and the Discovery of a Sensitivity Failure (Day 3)

**Files:**
- Create: `pilot/report/chapters/day03_baseline_inference.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day03_baseline_inference}`)

**Interfaces:**
- Consumes: `\citep{xiao2024florence2}` (Florence-2's task vocabulary, referenced again).
- Produces: Diagram B (TikZ, proxy v1 vs. v2 side-by-side) — the v2 logic (`all_boxes_covered`, proximity thresholds) is referenced again in Chapters 5, 9, 10 as "the person-relative redesign's fallback branch."

This is one of the richest chapters (full `day3_findings.md` + two commits). Target ~5-6 pages.

**Source material:**
- Florence-2-base-ft has **no native free-form VQA task token** — task vocabulary is grounding/captioning tokens (`<CAPTION>`, `<OD>`, `<OPEN_VOCABULARY_DETECTION>`, `<CAPTION_TO_PHRASE_GROUNDING>`). Each safety rule was reframed as a grounding query, not a yes/no question.
- `inference.py`'s `answer_rule()` dispatches: `rule_4` → `_answer_rule_4` (proximity check between independently grounded `"worker"` and `"excavator"`); rules 1-3 → `_answer_presence_rule`.
- `scripts/03_run_baseline_inference.py` ran over all 163 samples → `results/baseline_predictions.csv` + 20 overlay images in `results/figures/baseline/`.
- Pipeline mechanics validated by visual inspection: `<OPEN_VOCABULARY_DETECTION>` returns plausible, tightly-localized boxes (e.g. image 7: a hard-hat box landing correctly on a worker's head); rule_4's proximity check (image 120) correctly grounds both "worker" and "excavator" with a sound spatial relationship. Mean inference time 174ms/call, mean confidence (mean token probability) 0.58 — consistent with Day 1's single-sample numbers. All 163 target `image_id`s found; `assigned_rule_id` round-trips correctly.
- **v1 (scene-level) sensitivity/specificity** — quantified against ground truth `primary_class`: presence-based proxy (rules 1-3): **3.0% sensitivity** (3/100 violations correctly flagged), 97.4% specificity (37/38 compliant correctly flagged); proximity-based proxy (rule 4): 61.5% sensitivity (8/13), 41.7% specificity (5/12).
- Root cause (not a code bug — boxes are real, well-localized detections, confirmed visually): the dataset's `rule_1_violation` etc. marks that **a specific worker** lacks PPE; `<OPEN_VOCABULARY_DETECTION>` answers a **scene-level existence** question ("does a hard hat appear anywhere?"). Most ConstructionSite images contain multiple workers; if even one worker is compliant, the existence query returns a box and the proxy says "compliant," even when a different worker in the same frame is the one the dataset flagged.
- Concrete traced case: image 79 (ground truth `fall_hazard`/rule_3, no edge protection) — the "guardrail" query matched the wooden excavation-pit formwork as a guardrail-shaped structure, so the proxy answered "compliant" even though the dataset's actual violation is a missing barrier elsewhere in the same scene.
- The rule_4 proximity proxy does **not** share this failure mode (relational, not existential) and already showed real, balanced discriminative signal (62%/42%) at this small n.
- **Decision, stated explicitly as a decision rather than silently patched**: redesigned the rules 1-3 proxy to a **person-relative** check. `_answer_presence_rule` now detects all `"worker"` boxes and all safety-object boxes independently, then flags **violation** if any detected worker has no nearby/overlapping safety-object box (`regions.all_boxes_covered`) — mirroring the dataset's own semantics that one non-compliant worker is enough to mark the image violated. Worn PPE (rules 1-2) uses a tight proximity threshold (`PRESENCE_PROXIMITY_FRACTION`: 0.08, i.e. 8% of max image dimension, since the object must be on the worker's body); rule 3's guardrail is structural and can protect from further away, so it uses a looser threshold (0.20, 20%). Images with zero detected workers fall back to the old scene-level existence check (12/163 samples hit this fallback).
- **v2 (person-relative) results** (re-ran `scripts/03_run_baseline_inference.py` on all 163 samples): presence-based proxy sensitivity jumps to **27.0%** (27/100), specificity falls to 73.7% (28/38) — **a 9x sensitivity improvement**, an expected precision/recall tradeoff for a proxy that now actually discriminates instead of defaulting to "compliant." Per-rule sensitivity: rule_1 (hard hat) 16% (8/50), rule_2 (harness) 30.8% (4/13), rule_3 (guardrail) 40.5% (15/37) — rule_3 benefits most, likely because its looser proximity threshold is closer in spirit to the relational rule_4 check that already worked well.
- **Residual, documented limitation**: Florence-2's `<OPEN_VOCABULARY_DETECTION>` for the singular phrase `"worker"` returns only **one** box even in multi-worker scenes — visually confirmed on image 7 (5 workers visible, ground truth `ppe_violation`) and image 79 (multiple people visible, ground truth `fall_hazard`). When the single detected worker happens to be the compliant one, the per-worker check still can't catch the actual violator. Caps how much further the person-relative redesign can improve sensitivity without a different detection strategy (e.g. `<DENSE_REGION_CAPTION>` for exhaustive instance enumeration) — explicitly not pursued in this pass, since the 9x gain already achieved is a meaningful, honestly-earned improvement.
- **Implication flagged forward** (to Chapters 5, 9, 10): Descriptive Accuracy, Robustness, and Bounded Completeness all assume the Day 3 baseline answer is a meaningful judgment to test the explanation against; for rules 1-3 under the *old* v1 proxy this assumption was nearly false (a near-constant "compliant" classifier has nothing for masking to flip) — this is exactly why the redesign happened before those later days ran, not after.

**Worked example figures:**
- `\includegraphics{baseline/0000007_rule_1.png}` — hard-hat box correctly landing on a worker's head, 5 workers visible in the original scene (the case that motivates the "only one worker box returned" residual limitation).
- `\includegraphics{baseline/0000120_rule_4.png}` — rule 4's proximity check, worker + excavator both grounded with a sound spatial relationship.

**Standard vs. non-standard:** Standard = treating VQA as a closed-vocabulary grounding/detection proxy task, a known workaround pattern for VLMs lacking a native VQA head. Non-standard = the person-relative redesign itself — a domain-specific adaptation mirroring the dataset's own per-worker semantics, not a textbook XAI technique; report this as the chapter's central methodological contribution.

**Diagram B — TikZ code (proxy v1 vs. v2, side-by-side; caption "Safety-rule proxy logic before (left) and after (right) the Day 3 redesign."):**

```latex
\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.1cm]
\node[block] (img1) {Image};
\node[block, below=of img1] (ovd1) {\texttt{<OVD>}\\(safety object)};
\node[block, below=of ovd1] (exist1) {Box exists\\anywhere?};
\node[block, below=of exist1, fill=rulegray] (ans1) {Answer:\\compliant / violation};
\draw[arrow] (img1) -- (ovd1);
\draw[arrow] (ovd1) -- (exist1);
\draw[arrow] (exist1) -- (ans1);
\node[diagcaption, above=0.1cm of img1] {\textbf{v1: scene-level} (3.0\% sensitivity)};

\node[block, right=3.2cm of img1] (img2) {Image};
\node[block, below left=1.1cm and 0.3cm of img2] (worker2) {\texttt{<OVD>}\\("worker") $\to$\\worker\_boxes};
\node[block, below right=1.1cm and 0.3cm of img2] (obj2) {\texttt{<OVD>}\\(safety object) $\to$\\object\_boxes};
\node[block, below=2.3cm of img2] (cover2) {all\_boxes\_covered\\(per-worker check)};
\node[block, below=of cover2, fill=rulegray] (ans2) {Answer:\\compliant / violation};
\draw[arrow] (img2) -- (worker2);
\draw[arrow] (img2) -- (obj2);
\draw[arrow] (worker2) -- (cover2);
\draw[arrow] (obj2) -- (cover2);
\draw[arrow] (cover2) -- (ans2);
\node[diagcaption, above=0.1cm of img2] {\textbf{v2: person-relative} (27.0\% sensitivity, 9x)};
\end{tikzpicture}
\caption{Safety-rule proxy logic before (left) and after (right) the Day 3 redesign.}
\label{fig:diagram-b}
\end{figure}
```

- [ ] **Step 1: Write `day03_baseline_inference.tex`**

Structure: `\chapter{Baseline Inference and the Discovery of a Sensitivity Failure: From Scene-Level to Person-Relative Grounding (Day 3)}`, sections `Motivation`, `Implementation`, `Architecture Before and After` (Diagram B), `Worked Examples` (the two figures), `Results` (the sensitivity/specificity tables, v1 and v2), `Problems Encountered and Resolutions` (the root-cause trace on image 79), `Standard vs. Non-Standard Choices`, `Limitations` (the single-worker-box residual gap), `Implications for the Paper` (frame the 3.0%→27.0% redesign as the chapter reviewers will most want explained; recommend a dedicated "Threats to Validity" paragraph on the residual detection gap).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day03_baseline_inference}` after `\include{day02_sample_selection}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; both images render (check `main.log` for `! LaTeX Error: File ... not found` — would indicate a wrong `\graphicspath`-relative filename); Diagram B renders without TikZ errors; bibliography section now shows at least 1 entry (`xiao2024florence2`).

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day03_baseline_inference.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 3: baseline inference and the proxy redesign (Day 3)"
```

---

## Task 6: Chapter 4 — Standardizing Explanation Regions for Masking-Based Tests (Day 4)

**Files:**
- Create: `pilot/report/chapters/day04_explanation_regions.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day04_explanation_regions}`)

**Interfaces:**
- Produces: Diagram C (TikZ, regions/masking layer) — `standardize_regions`'s area-ranking choice is referenced again in Chapters 6, 7, 10 as the mechanism behind their PPE-disadvantage findings.

**Source material:**
- `regions.py`: `Region` dataclass (`box`, `source: "model"|"grid"`, `label`); `standardize_regions(boxes, labels, image_size, grid=(4,4))` ranks model-returned boxes by **area, descending** — an explicit stand-in for a real saliency score, which Florence-2's grounding output does not provide; falls back to an unranked 4×4 grid **only** when zero boxes were returned at all.
- Also added: `iou()`, `mask_region()` (black-fill or Gaussian-blur fill), `grid_fallback_regions()`, `boxes_overlap_or_close()`, `all_boxes_covered()` (this last one is what Chapter 3's redesign actually calls — note explicitly that Day 3 and Day 4's work were developed together, per commit `6de76fb`).
- `scripts/04_extract_regions.py` reuses Day 3's already-computed boxes (**no new GPU inference**) to rank regions for all 163 samples; saves 10 before/after masked-image pairs for visual sanity-checking.
- **Result**: 159/163 samples (97.5%) used a real model box as the top region; 4/163 (2.5%) correctly fell back to the grid. Visual inspection of 2 pairs confirmed `mask_region` blacks out exactly the ranked top box and nothing else.
- Problems & resolutions: no real saliency signal exists from Florence-2's grounding output → explicit design decision to use box area as the ranking heuristic, flagged plainly as a stand-in, not a claim of true importance. **State explicitly that this choice is later shown (Chapters 6, 7, 10) to systematically disadvantage small PPE objects** — forward-reference this now so the limitation reads as foreseen, not discovered by accident.
- Standard vs. non-standard: standard = bounding-box masking/occlusion as an explanation-testing primitive (a long-established vision-XAI pattern, e.g. occlusion sensitivity maps); non-standard = ranking purely by area as a substitute for an importance score — an explicit, acknowledged simplification, not a published method.
- Implications for the paper: flag the area-ranking heuristic explicitly as a limitation/future-work item — it is the single highest-leverage fix identified across the whole pilot (see Chapter 10 Finding 1 and Chapter 14's task list item 3).

**Worked example figures:**
- `\includegraphics{regions/0000007_rule_1_before.png}` and `\includegraphics{regions/0000007_rule_1_after.png}` — before/after masked pair (use `subcaption`, side by side, same image as Chapter 3's worked example for narrative continuity).

**Diagram C — TikZ code (regions/masking layer; caption "Regions/masking layer added on top of Day 3's baseline grounding output."):**

```latex
\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.6cm]
\node[block] (boxes) {Baseline boxes\\+ labels (Day 3)};
\node[block, right=of boxes] (rank) {standardize\_regions\\(rank by area /\\grid fallback)};
\node[block, right=of rank] (region) {Ranked\\Region list};
\node[block, right=of region] (mask) {mask\_region\\(black / blur)};
\node[block, right=of mask] (out) {Masked image\\for re-inference};
\draw[arrow] (boxes) -- (rank);
\draw[arrow] (rank) -- (region);
\draw[arrow] (region) -- (mask);
\draw[arrow] (mask) -- (out);
\end{tikzpicture}
\caption{Regions/masking layer added on top of Day 3's baseline grounding output.}
\label{fig:diagram-c}
\end{figure}
```

- [ ] **Step 1: Write `day04_explanation_regions.tex`**

Structure: `\chapter{Standardizing Explanation Regions for Masking-Based Tests (Day 4)}`, sections `Motivation`, `Implementation`, `Architecture: A New Masking Layer` (Diagram C), `Worked Example` (before/after figure pair), `Results` (159/163 vs. 4/163), `Problems Encountered and Resolutions` (no-saliency-signal decision), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (forward-reference the PPE-disadvantage finding). Keep to ~2-3 pages per the design's "scale to evidence" rule (Day 4 has no dedicated `findings.md`; do not pad beyond what's listed above).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day04_explanation_regions}` after `\include{day03_baseline_inference}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; the before/after figure pair renders side by side; Diagram C renders without TikZ errors.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day04_explanation_regions.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 4: standardized explanation regions (Day 4)"
```

---

## Task 7: Chapter 5 — Descriptive Accuracy: Does Masking the Top Region Change the Answer? (Day 5)

**Files:**
- Create: `pilot/report/chapters/day05_descriptive_accuracy.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day05_descriptive_accuracy}`)

**Interfaces:**
- Consumes: `\citep{arreche2024exai}` (descriptive accuracy is one of the six metrics defined there).
- Produces: nothing new (no diagram — composes existing `inference.py` + `regions.py`, no new pipeline stage).

Rich chapter (full `day5_findings.md`). Target ~4-5 pages.

**Source material:**
- `descriptive_accuracy.py`'s `evaluate()`: masks top-1 region (black-fill), reruns `answer_rule`, compares to baseline; then masks top-1+top-2 together, reruns again.
- `scripts/05_descriptive_accuracy.py` ran over 163 samples → `results/descriptive_accuracy.csv` + 10 flip-example overlays.
- **Results**: top-1 masking flip rate **36.2%**; top-1+top-2 **35.0%** — lower, a non-monotonic anomaly worth its own subsection.
- By `primary_class` (top-1): `struck_by_risk` 61.5%, `fall_hazard` 46.0%, `ppe_violation` 28.0%, `compliant` 28.0%.
- By `assigned_rule_id` (top-1): rule_4 44.0%, rule_3 42.9%, rule_2 38.5%, rule_1 27.0%. These track Day 3's redesigned-proxy sensitivity ordering closely — not coincidental: a proxy that barely discriminates also has little for masking to disrupt.
- **Finding — masking top-2 is not strictly stronger than masking top-1**: naively, masking *more* evidence should never un-flip an answer top-1 alone already flipped; here it does, in 15/163 samples. Traced concretely on image `0000069` (rule_1): baseline boxes are one worker box (area 10,778 px²) and one hard-hat box (area 891 px²) — the worker box is larger, so Day 4's area-based ranking puts it at top-1. Masking top-1 (the worker) makes the rerun unable to detect that worker at all (`worker_boxes` comes back empty); `_answer_presence_rule` falls back to its scene-level branch ("no worker detected, so compliant if the object exists anywhere"); the hard-hat box is still visible (only the worker was masked), so the fallback answers **compliant** — a flip from baseline's `violation`. Also masking top-2 (the hard hat) means neither region is visible; the same fallback branch now finds no object either, so it answers **violation** — reverting to the original baseline. 14 of the 15 regressions follow this exact full-revert pattern.
- This is a real interaction between the masking test and the person-relative redesign's own fallback logic (Chapter 3), **not a bug in either component alone**: masking the *worker* region doesn't just remove evidence, it removes the precondition (`if not worker_boxes`) the per-worker check needs to run at all, silently rerouting to the cruder scene-level branch.
- **Implication flagged forward** (to Chapters 9, 10): Robustness and Bounded Completeness both rerun `answer_rule` on modified images and compare to baseline — the same shape of test as this chapter. The same fallback-rerouting risk applies there.

**Worked example figures:**
- `\includegraphics{descriptive_accuracy/0000069_rule_1_flip.png}` — top-1 region is the worker box; masking it removes that worker from detection entirely (the fallback-rerouting case).
- `\includegraphics{descriptive_accuracy/0000120_rule_4_flip.png}` — top-1 region is the excavator box itself; masking it removes the proximity check's other half, flipping `hazard`→`safe`. State explicitly: "this one *is* the intuitive case — the masked region is exactly the rule-defining object, and removing it removes the hazard signal in a way that matches what 'explanation' should mean."

**Standard vs. non-standard:** Standard = deletion/masking-based descriptive accuracy, a well-established explanation-fidelity pattern (cite `\citep{arreche2024exai}` as the direct methodological source for this metric). Non-standard/contribution = discovering and mechanistically explaining the non-monotonic "masking more un-flips the answer" anomaly as a fallback-branch artifact rather than dismissing it as noise.

- [ ] **Step 1: Write `day05_descriptive_accuracy.tex`**

Structure: `\chapter{Descriptive Accuracy: Does Masking the Top Region Change the Answer? (Day 5)}`, sections `Motivation`, `Implementation`, `Results` (the headline table + by-class/by-rule breakdowns), `Worked Examples` (both figures), `The Non-Monotonicity Anomaly` (the traced `0000069` case, mechanism explanation), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (recommend the non-monotonicity finding as a generalizable diagnostic: "watch for non-monotonic masking results as a sign of hidden fallback logic," applicable beyond this one pilot).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day05_descriptive_accuracy}` after `\include{day04_explanation_regions}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; both figures render; bibliography now includes `arreche2024exai`.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day05_descriptive_accuracy.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 5: descriptive accuracy and the non-monotonicity finding (Day 5)"
```

---

## Task 8: Chapter 6 — Visual Sparsity via Decoder-Encoder Cross-Attention (Day 6)

**Files:**
- Create: `pilot/report/chapters/day06_visual_sparsity.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day06_visual_sparsity}`)

**Interfaces:**
- Consumes: `\citep{abnar2020rollout}` (the method explicitly NOT used, with reasoning why), `\citep{arreche2024exai}` (sparsity as one of the six metrics), optionally `\citep{kokhlikyan2020captum,bach2015lrp,shrikumar2017deeplift}` when discussing attribution alternatives considered.
- Produces: Diagram D (TikZ, attribution layer) — `attribution.py`'s cross-attention extraction is referenced again in Chapter 14's open question 1 (to Prof. Abdallah, on attribution-method acceptability).

Richest chapter alongside Chapter 9 (full `day6_findings.md`). Target ~5-6 pages.

**Source material:**
- The plan called for `attention_rollout()` first. Before implementing, Florence-2's actual cached modeling code (`modeling_florence2.py`) was checked rather than assuming a textbook ViT-style rollout would apply, and it doesn't cleanly: Florence-2's encoder runs **one shared self-attention stack over a fused image+text sequence**: `[1 global "spatial_avg_pool" token] + [576 image-patch tokens, 24×24] + [text prompt tokens]`. Verified directly — `_encode_image` returns `torch.cat([spatial_avg_pool_x, temporal_avg_pool_x])` (1+576=577 tokens, confirmed by printing `image_features.shape`), and 576=24×24 follows from the DaViT vision tower's four conv stages (strides 4,2,2,2) applied to the processor's fixed 768×768 input (768→192→96→48→24, computed by hand).
- Classic attention rollout \citep{abnar2020rollout} is defined for a self-attention-only stack processing **one** modality. Rolling out Florence-2's fused encoder would recursively mix in text-token attention at every layer in a way the method was never designed for, and the mixing isn't separable back out after the fact.
- **Used decoder→encoder cross-attention instead**: `attribution.py`'s `cross_attention_heatmap()` — for each generated token, the decoder attends back to the full encoder sequence; averaged over decoder layers, heads, and generated tokens; the global token (index 0) is dropped before reshaping the remaining 576 values into the 24×24 patch grid.
- **Off-by-one trap, flagged explicitly**: 577 is not a perfect square; assuming `sqrt(num_image_tokens)` without checking would silently misalign every heatmap by one cell.
- **Decoding-mode difference from Chapters 3-5**: cross-attention extraction always uses `num_beams=1` (greedy), not the `num_beams=3` beam search used for the logged answers, because HF's beam search reorders the batch dimension at every step and `generate()`'s returned attentions aren't automatically un-reordered. Spot-checked on image `0000007`: the greedy boxes matched Chapter 4's beam-search boxes closely in position — the substitution doesn't change *what* gets attributed.
- **Results**: `scripts/06_visual_sparsity.py` ran over 163 samples (4/163 grid-fallback cases excluded — no grounded phrase to attribute attention to). Mean over 159 scored samples: top-5-of-576-cells mass ratio **0.108**; top-20 mass ratio 0.247; cells above 0.5 threshold: 7.0.
- By rule (top-5 mass ratio): rule_1 0.125, rule_2 0.113, rule_3 0.098, rule_4 0.079.
- By attributed phrase (top-5 mass ratio): **hard hat 0.195** (most concentrated), safety harness 0.118, worker 0.115, guardrail 0.099, **excavator 0.079** (least concentrated).
- **Finding — concentration is driven mainly by target physical size, not "explanation quality"**: hard hat is ~2.5× more concentrated than excavator. Mean box area vs. concentration: excavator 287,913 px² → 0.079; guardrail 310,291 px² → 0.099; safety harness 178,491 px² → 0.118; worker 79,005 px² → 0.115; hard hat 94,330 px² → 0.195. Correlation between log(box area) and top-5 mass ratio across all scored samples: **−0.70**. This is a real confound, not a bug: a small object's 576-cell grid mechanically has fewer cells it *could* cover, concentrating attention mass regardless of whether the underlying reasoning is any good.
- **Design decision carried forward**: visual and text attribution streams are reported **separately**, never merged into one score — a decision later mirrored by Chapter 9's answer-level vs. explanation-level robustness split.

**Worked example figures (side by side via `subcaption`):**
- `\includegraphics{sparsity/0000023_rule_1_hard_hat.png}` — small, sharp hotspot on a distant worker's hard hat, the clearest "focused" example.
- `\includegraphics{sparsity/0000079_rule_3_guardrail.png}` — attention spread broadly across most of the frame, the clearest "scattered" example, consistent with the guardrail's large structural footprint.

**Standard vs. non-standard:** Standard = decoder→encoder cross-attention as an attribution signal for encoder-decoder transformers (well-established for, e.g., machine-translation attention visualization). Non-standard = the explicit, verified decision **not** to use attention rollout, the textbook ViT XAI method the plan originally specified, because Florence-2's architecture breaks rollout's single-modality assumption — a genuine, documented methodological substitution.

**Diagram D — TikZ code (attribution layer; caption "Attribution layer: decoder$\to$encoder cross-attention, substituted for attention rollout."):**

```latex
\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.5cm]
\node[block] (gen) {Image + prompt\\$\to$ decoder token $t$\\(greedy, num\_beams=1)};
\node[block, right=of gen] (cross) {Cross-attention\\to 577-token\\encoder sequence};
\node[block, right=of cross] (avg) {Average over\\layers, heads,\\generated steps};
\node[block, below=of avg] (drop) {Drop global token,\\reshape 576 $\to$ 24$\times$24};
\node[block, left=of drop] (norm) {Normalize\\to $[0,1]$};
\node[block, left=of norm] (metric) {Sparsity metrics:\\topk\_mass\_ratio,\\regions\_above\_threshold};
\draw[arrow] (gen) -- (cross);
\draw[arrow] (cross) -- (avg);
\draw[arrow] (avg) -- (drop);
\draw[arrow] (drop) -- (norm);
\draw[arrow] (norm) -- (metric);
\end{tikzpicture}
\caption{Attribution layer: decoder$\to$encoder cross-attention, substituted for attention rollout.}
\label{fig:diagram-d}
\end{figure}
```

- [ ] **Step 1: Write `day06_visual_sparsity.tex`**

Structure: `\chapter{Visual Sparsity via Decoder-Encoder Cross-Attention: Why Attention Rollout Does Not Apply (Day 6)}`, sections `Motivation`, `Method Deviation from the Plan` (the verified architecture check, citing `\citep{abnar2020rollout}` for what was not used and why), `Architecture: An Attribution Layer` (Diagram D), `Implementation Details` (off-by-one trap, decoding-mode difference), `Results` (the tables), `Worked Examples` (the two side-by-side figures), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (frame as a candidate dedicated subsection on adapting attribution to fused-modality encoders; cite `\citep{kokhlikyan2020captum,bach2015lrp,shrikumar2017deeplift}` as other attribution families considered and also inapplicable to this architecture as-is, tying to Chapter 14's open question 1).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day06_visual_sparsity}` after `\include{day05_descriptive_accuracy}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; Diagram D renders (note the `drop`/`norm`/`metric` nodes are laid out in a second row — check for node-overlap visually in the PDF, adjust `node distance`/`below=of` offsets if labels collide); both sparsity figures render side by side; bibliography includes `abnar2020rollout`.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day06_visual_sparsity.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 6: visual sparsity via cross-attention, not rollout (Day 6)"
```

---

## Task 9: Chapter 7 — Stability Under Stochastic Decoding (Day 7)

**Files:**
- Create: `pilot/report/chapters/day07_stability.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day07_stability}`)

**Interfaces:**
- Consumes: `\citep{arreche2024exai}` (stability as one of the six metrics).
- Produces: nothing new (no diagram; reuses existing `answer_rule` + region ranking).

Rich chapter (full `day7_findings.md`). Target ~4-5 pages.

**Source material:**
- Plan's own warning, taken seriously: Florence-2's default decoding (beam search, no sampling) is deterministic — three literal reruns would be bit-identical and "100% stable" by construction, proving nothing.
- `metrics/stability.py`'s `run_n_times()`: reruns each sample 3× with `do_sample=True, num_beams=1, temperature=0.7` instead.
- **Confirmed non-vacuous before trusting any aggregate**: mean `answer_agreement_rate` across all 163 samples is **0.775**, not 1.0 — roughly a quarter of rerun pairs genuinely disagree. The deterministic beam-search answer (already logged in `baseline_predictions.csv`) is reported separately as `deterministic_matches_majority` (83.4%), not folded into the stability score.
- **Results table**: `answer_agreement_rate` 0.775; `deterministic_matches_majority` 83.4%; `top_region_overlap_score` (IoU of the area-ranked top-1 region) 0.667; `object_region_overlap_score` (IoU of the rule's actual safety-object box specifically) 0.602.
- By rule: rule_1 (hard hat) 0.735 agreement / 0.499 object IoU; rule_2 (harness) 0.744 / 0.533; rule_3 (guardrail) 0.782 / 0.615; rule_4 (excavator) 0.893 / 0.922.
- **Finding — `top_region_overlap_score` hides how unstable the safety-relevant box actually is**: 0.667 mean looks reasonably stable, but it's computed on the area-ranked top-1 region, which for rules 1-3 is almost always the **worker** box (much larger than the PPE item), not the hard hat/harness/guardrail the rule is actually about. `object_region_overlap_score`, computed specifically on `object_boxes[0]`, exposes a large gap: rule_1's hard-hat box overlaps only **0.499** across reruns vs. rule_4's excavator box at **0.922** — nearly 2× more stable.
- Ruled out an easy alternative explanation first: could this be multiple detected boxes simply getting reordered across stochastic runs (so `object_boxes[0]` picks a *different* real box, not a *moved* one)? Checked directly — only 4-8% of samples have more than one object box at all (rule_1 6.3%, rule_2 3.8%, rule_3 8.2%, rule_4 4.0%), so reordering can't explain a systematic ~2× gap.
- Real driver: a combination of object size **and shape**, parallel to Chapter 6's box-area confound. Median box area vs. `object_region_overlap_score`: hard hat 570 px² → 0.499; harness 7,169 px² → 0.533; guardrail 221,163 px² → 0.615; excavator 215,069 px² → 0.922. Size alone doesn't fully explain it — guardrail and excavator have almost identical median area (221k vs. 215k px²) but very different stability (0.615 vs. 0.922); the remaining gap looks like shape/rigidity — a guardrail is long, thin, sometimes segmented, so small boundary shifts move its IoU noticeably; an excavator is one compact, rigid, visually distinctive object.
- **Implication flagged for Chapter 11**: both this chapter and Chapter 6 show the same underlying pattern — metrics computed on small, body-worn PPE items look systematically worse than metrics on large rigid objects, for reasons that are at least partly mechanical, not purely about explanation quality. Worth stating the irony plainly: hard hat and harness are exactly the PPE items where a *stable* explanation matters most for real-world trust, and they are the least stable by this measure.

**Worked example figures (side by side):**
- `\includegraphics{stability/0000023_rule_1_reruns.png}` — the overlaid top-1 boxes (worker, large) sit almost exactly on top of each other across all 3 reruns (the worker detection is stable); the hard-hat box itself, not drawn in this overlay since it loses the area-ranking, is what actually varies (`object_region_overlap_score=0.069` for this sample).
- `\includegraphics{stability/0000079_rule_3_reruns.png}` — the guardrail/formwork box is nearly identical across all 3 reruns (`object_region_overlap_score=0.955`).

**Standard vs. non-standard:** Standard = sampling-based stability/consistency testing, rooted in `\citep{arreche2024exai}`'s six-metric framework. Non-standard/contribution = introducing `object_region_overlap_score` as a **separate** metric from the naive top-1-region IoU, specifically to unmask a confound the naive metric hides — a methodological refinement worth keeping in the framework going forward.

- [ ] **Step 1: Write `day07_stability.tex`**

Structure: `\chapter{Stability Under Stochastic Decoding (Day 7)}`, sections `Motivation` (the vacuous-result warning), `Implementation`, `Results` (both tables), `Finding: A Hidden Instability` (the `object_region_overlap_score` investigation, ruling out reordering, the size+shape explanation), `Worked Examples` (both figures), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (recommend always reporting "the metric's own target object" stability alongside whatever a ranking heuristic happens to pick — generalizable guidance, tie explicitly to Chapter 4's area-ranking limitation).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day07_stability}` after `\include{day06_visual_sparsity}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; both stability figures render side by side.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day07_stability.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 7: stability under stochastic decoding (Day 7)"
```

---

## Task 10: Chapter 8 — Efficiency and the Cost of Six Metrics at Scale (Day 8)

**Files:**
- Create: `pilot/report/chapters/day08_efficiency.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day08_efficiency}`)

**Interfaces:**
- Consumes: `\citep{arreche2024exai}` (efficiency as one of the six metrics).
- Produces: nothing new (pure measurement layer, no new pipeline stage).

Rich chapter (full `day8_findings.md`). Target ~4 pages.

**Source material:**
- **Instrumentation gap discovered, not assumed away**: the plan expected `aggregate_timings()` to pull `inference_ms` straight from Days 3/5/6/7's CSVs with zero new model calls; checking the actual headers first showed only `baseline_predictions.csv` (Day 3) ever logged it — `descriptive_accuracy.csv`, `visual_sparsity.csv`, `stability.csv` have no timing column at all.
- Handled each gap honestly rather than inventing numbers: Day 5 reruns the *exact same call type* Day 3 already timed (`answer_rule`, `num_beams=3`) → its cost is recovered exactly, no new model calls, as `rerun_count × that sample's own inference_ms` (`rerun_count` from `region_extraction.csv`'s `n_regions` column). Days 6 and 7 use two decoding configs never timed before (`output_attentions=True` forces eager attention — slower; `do_sample=True, num_beams=1` — no beam multiplicity) → ran a small, fresh **15-sample calibration** for each, a deliberate bounded deviation, documented rather than hidden.
- **Results table**: Day 3 baseline inference — 1 `answer_rule` call (2 generate calls inside), mean 271ms/sample, extrapolated 4.5 min per 1,000 samples. Day 5 descriptive-accuracy reruns — 1.94 calls, 526ms/sample, 8.8 min/1,000. Day 6 cross-attention attribution — 1 call (15-sample calibration), 215ms/sample, 3.6 min/1,000. Day 7 stability sampled reruns — 3 calls, 778ms/sample, 13.0 min/1,000.
- **Full 6-metric pipeline at n=1,000 (linear extrapolation): ~0.50 GPU-hours (~30 minutes) on a single RTX 3070.** Practically trivial — not a blocker for scaling to the full 3,004-image test split.
- Day 4 (region extraction) and Day 11 (analysis) do **no model inference at all** — pandas/PIL operations on already-computed CSVs, correctly excluded from the table.
- **Finding 1 — "ms per call" across days needs call-count normalizing**: Day 6's 215ms looks cheaper than Day 3's 271ms at face value, seemingly contradicting Day 6's own finding that `output_attentions=True` forces a slower eager-attention path. The catch: `answer_rule` (Days 3/5/7) makes **two** generate calls per "call" (worker phrase + object phrase), while `cross_attention_heatmap` (Day 6) makes **one**. Normalized to a single-generate-call basis: Day 3/7's implied per-call cost is ~271/2≈135ms and ~259/2≈130ms; Day 6's single call costs 215ms — **about 65% more per call**, consistent with the eager-attention overhead once the comparison is apples-to-apples.
- **Finding 2 — `num_beams=3` beam search is barely slower than `num_beams=1` sampling here**: Day 7's sampled calls (259ms/call) are only ~4% faster than Day 3's beam-search calls (271ms/call) — not the ~3× a naive "3 beams = 3× compute" intuition would predict. Likely reason: these grounding outputs are very short (~7-8 generated tokens), so latency is dominated by the *fixed* cost of encoding the image through the DaViT vision tower once per call, not the decode loop where beam multiplicity would matter. Beam count would likely matter far more for long-sequence tasks (e.g. `<MORE_DETAILED_CAPTION>`).
- Sanity check on the extrapolation: Day 3/5's numbers are exact sums over the real 163 samples (44.2s, 85.7s) — no extrapolation error possible there. Day 6/7's numbers rest on a 15-sample calibration scaled linearly; the calibrated 778ms/sample for Day 7 implies ~127s of pure GPU compute for all 163×3 reruns, the right order of magnitude vs. what was actually observed running `07_stability_test.py` end-to-end.
- **Caveat carried to Chapter 11/12**: Day 6/7's per-sample costs rest on a 15-sample calibration, not the full 163-sample run, unlike Day 3/5's exact aggregation — good enough for an order-of-magnitude story, should be re-measured directly (instrumented `inference_ms`) before quoting precise numbers in anything more formal.

**Worked example figure:**
- `\includegraphics{efficiency/per_sample_cost.png}` — per-sample timing breakdown behind the headline numbers.

**Standard vs. non-standard:** Standard = wall-clock timing aggregation and linear extrapolation for compute-budget estimation, per `\citep{arreche2024exai}`'s efficiency metric. Non-standard/notable = the normalized-per-generate-call comparison needed to make cross-day timing comparisons apples-to-apples — an easy-to-miss methodological detail worth stating explicitly in any paper's efficiency-methodology paragraph.

- [ ] **Step 1: Write `day08_efficiency.tex`**

Structure: `\chapter{Efficiency and the Cost of Six Metrics at Scale (Day 8)}`, sections `Motivation`, `An Instrumentation Gap, Handled Honestly` (the calibration decision), `Results` (the table + the two findings), `Worked Example` (the figure), `Sanity Check on the Extrapolation`, `Standard vs. Non-Standard Choices`, `Implications for the Paper` (frame efficiency as the strong, low-controversy "good news" result section — compute is conclusively not an obstacle to scaling).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day08_efficiency}` after `\include{day07_stability}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; the efficiency figure renders.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day08_efficiency.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 8: efficiency at scale (Day 8)"
```

---

## Task 11: Chapter 9 — Robustness to Perturbation and Prompt Rewording (Day 9)

**Files:**
- Create: `pilot/report/chapters/day09_robustness.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day09_robustness}`)

**Interfaces:**
- Consumes: `\citep{arreche2024exai}` (robustness as one of the six metrics).
- Produces: Diagram E (TikZ, perturbation layer) — the answer-level-vs-explanation-level split is referenced again in Chapters 10, 11, 14 as the pilot's central recurring methodological theme.

**This is the richest chapter in the pilot** (full `day9_findings.md`, 169 lines, the most detailed findings doc). Target ~6-7 pages — the longest chapter.

**Source material:**
- `perturbations.py`: `blur` (Gaussian, `ksize=8.0`), `low_light` (gamma correction, `gamma=2.5`), `occlude` (centered black square, `frac=0.2` of area), `contrast_shift` (factor=0.3), `paste_patch` (synthetic patch compositing, used only by the stretch test).
- `robustness.py`'s `RobustnessResult`: `answer_changed`, `confidence_drop`, `object_box_iou`, `worker_lost` — `worker_lost` flags specifically when the baseline detected ≥1 worker but the perturbed rerun detected none, because Chapter 5 already found masking the worker reroutes `_answer_presence_rule` into a degenerate fallback.
- `scripts/09_robustness_test.py`: 163 samples × 4 Level-1 perturbations (652 reruns) + Level-2 reworded-prompt reruns (138 of 163; rule_4 excluded, no second phrasing, logged with `excluded_reason="rule_4_has_no_second_phrasing"`, not scored as 0% change — caught before writing any aggregation code) + a 20-sample synthetic hard-hat-patch stretch test. All reruns use the **same deterministic decoding as baseline** (`num_beams=3`, no sampling) so perturbation sensitivity isn't confounded with Chapter 7's sampling-noise sensitivity.
- **Headline `answer_changed` rates by perturbation** (full table, with `worker_lost`/`confidence_drop`/`object_box_iou`): occlude 28.2% (worker_lost 8.6%, confidence_drop −0.012 [confidence rose], object_box_iou 0.530); reworded_prompt 27.5% (0.0%, −0.053 [rose], 0.333); blur 27.0% (0.0%, 0.013, 0.287); low_light 11.7% (0.6%, 0.011, 0.816); contrast_shift 10.4% (0.0%, 0.008, 0.825). Brightness/contrast are the mildest perturbations; blur, occlusion, and prompt rewording are all substantially more disruptive — but for **three different reasons**, not one.
- **Finding 1 — occlude's instability is partly the Chapter-5-flagged fallback artifact, not genuine sensitivity**: of occlude's 46 answer-flip cases, 9 (20%) also have `worker_lost=True`. `worker_lost` rate under occlude is uneven across rules: rule_2 19.2% ≫ rule_4 16.0% > rule_3 8.2% ≫ rule_1 1.6%. Traced concretely on image `0000341` (rule_3, baseline "compliant"): **both baseline detections were already wrong** (the "worker" box is a false positive on the excavator's cab window; the "object"/guardrail-query box is the entire excavator silhouette) but happened to geometrically satisfy the coverage check. After occlusion, **both** detections vanish (`worker_lost=True`, `object_box_iou=NaN`) even though the occluded square doesn't fully cover either original box — occlusion disrupts Florence-2's whole-scene grounding context, not just the literal pixels under the square. The fallback then defaults to "violation" purely because no object box survived. Reads, in an aggregate table, identical to "the model correctly noticed the safety equipment became occluded" — **it isn't**; it's the same degenerate-fallback mechanism Chapter 5 already flagged, just triggered by a perturbation instead of a deliberate mask.
- **Finding 2 — a stable answer can hide a fully-drifted explanation**: across all 611 stable-answer rows with a comparable object box on both sides, **28.3% have `object_box_iou < 0.3`** — uneven by perturbation: blur 58.8%, reworded_prompt 56.7%, occlude 32.4%, low_light 5.7%, contrast_shift 2.8%. Traced concretely on image `0000034` (rule_2, baseline "compliant", `answer_changed=False`): the worker's entire baseline box sits **inside** the occluded square, yet the rerun still reports `worker_lost=False` (it found *a* worker box somewhere) and the final answer is unchanged — while `object_box_iou=0.005`, i.e. the object box moved to a near-completely different location. **An answer-level robustness number alone would call this sample perfectly robust; it isn't, at the explanation level.** This is the strongest single result in the pilot motivating reporting answer-level and explanation-level robustness as **two separate numbers**, never blended — mirroring Chapter 6's visual/text-sparsity separation.
- **Finding 3 — rule_2's two phrasings aren't synonyms**: Level-2 answer-change rate by rule: rule_2 (harness/lanyard) 50.0%, rule_1 (hard hat/vest) 31.7%, rule_3 (guardrail/edge barrier) 10.2%. rule_3's two phrasings are near-synonymous and grounded consistently; rule_2's are **not** really synonyms — a harness is a body garment, a lanyard is the tether attached to it — so the two queries can legitimately ground *different physical objects* in the same scene, not just one concept under different wording. A prompt-design caveat for future `prompts.py` revisions, not a model robustness failure.
- **Finding 4 — confidence rises, not falls, under occlude and reworded_prompt**: mean `confidence_drop` is *negative* for occlude (−0.012) and reworded_prompt (−0.053) — average confidence went **up** after the perturbation, despite both having among the highest answer-change rates. Confirms, again, the confidence proxy is uncalibrated, not a correctness signal.
- **Stretch test — synthetic hard-hat patch (20 `ppe_violation`/rule_1 candidates)**: caught a **vacuous-denominator issue** before reporting the headline number. 16/20 candidates already had a baseline proxy answer of "compliant" (a known sensitivity gap, not what this test measures), so "fooling" is undefined for them; the naive 15% (3/20) flip rate would have silently diluted the real signal. Restricting to the **4** candidates the proxy already called "violation" (the only ones where a flip means anything): **3/4 (75%) flipped to "compliant"** after the patch. `09_robustness_test.py` now prints both numbers explicitly. n=4 is too small to generalize, and the 4 traced cases show **no clean placement→flip correlation**: `0000233` (off-target, torso/shoulder, not head — crude `_head_box` approximation, no pose estimation) → flipped; `0000641` (on-target, squarely on the head) → did **not** flip; `0001431` (on-target) → flipped; `0001842` (on-target, scene already has many real hard-hat wearers) → flipped. A well-placed patch failed to fool the model while an off-target one succeeded — Florence-2's "hard hat" grounding appears generally imprecise about *where* on a person the cue needs to be, in both directions.

**Worked example figures (use `subcaption`, before/after pairs):**
- `\includegraphics{robustness/verify/0000341_before.png}` / `\includegraphics{robustness/verify/0000341_after_occlude.png}` — Finding 1's traced case.
- `\includegraphics{robustness/verify/0000034_before.png}` / `\includegraphics{robustness/verify/0000034_after_occlude.png}` — Finding 2's traced case (the single best image for "answer-level robustness alone is not enough").
- `\includegraphics{robustness/verify/0000233_patch_before.png}` / `\includegraphics{robustness/verify/0000233_patch_after.png}` and `\includegraphics{robustness/verify/0000641_patch_before.png}` / `\includegraphics{robustness/verify/0000641_patch_after.png}` — the stretch test's counterintuitive on-target/off-target pair.

**Standard vs. non-standard:** Standard = perturbation-based robustness testing (blur/noise/occlusion), same family as `\citep{arreche2024exai}`'s robustness metric. Non-standard/contribution = the explicit separation of answer-level vs. explanation-level robustness as two numbers, motivated by a directly-traced concrete failure case rather than asserted on principle; also the denominator-correction in the patch-stretch test.

**Diagram E — TikZ code (perturbation layer, three parallel branches; caption "Perturbation layer: three parallel robustness branches, Level 1/Level 2/patch-stretch."):**

```latex
\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.3cm]
\node[block] (img) {Original image};
\node[block, below left=1.2cm and -0.6cm of img] (l1) {blur / low\_light /\\occlude / contrast\_shift};
\node[block, below=1.2cm of img] (l2) {phrasing\_index=1\\(no image change)};
\node[block, below right=1.2cm and -0.6cm of img] (l3) {paste\_patch\\(head\_box, hard hat)};
\node[block, below=1.6cm of l1] (rerun1) {answer\_rule rerun};
\node[block, below=1.6cm of l2] (rerun2) {answer\_rule rerun};
\node[block, below=1.6cm of l3] (rerun3) {answer\_rule rerun};
\node[block, below=2.4cm of l2, fill=rulegray] (cmp) {Compare to baseline:\\answer\_changed, confidence\_drop,\\object\_box\_iou, worker\_lost};
\draw[arrow] (img) -- (l1);
\draw[arrow] (img) -- (l2);
\draw[arrow] (img) -- (l3);
\draw[arrow] (l1) -- (rerun1);
\draw[arrow] (l2) -- (rerun2);
\draw[arrow] (l3) -- (rerun3);
\draw[arrow] (rerun1) -- (cmp);
\draw[arrow] (rerun2) -- (cmp);
\draw[arrow] (rerun3) -- (cmp);
\end{tikzpicture}
\caption{Perturbation layer: three parallel robustness branches, Level 1/Level 2/patch-stretch.}
\label{fig:diagram-e}
\end{figure}
```

- [ ] **Step 1: Write `day09_robustness.tex`**

Structure: `\chapter{Robustness to Perturbation and Prompt Rewording: When a Stable Answer Hides a Drifted Explanation (Day 9)}`, sections `Motivation`, `Implementation`, `Architecture: A Perturbation Layer` (Diagram E), `Results` (the headline table), `Finding 1: Occlusion's Fallback Artifact` (with figure pair), `Finding 2: A Stable Answer Hiding a Drifted Explanation` (with figure pair — flag this as the chapter's central result), `Finding 3: Not All Phrasings Are Synonyms`, `Finding 4: An Uncalibrated Confidence Proxy`, `Stretch Test: A Synthetic Hard-Hat Patch` (with figure pairs, the denominator-correction story), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (this chapter's two traced case studies are the best candidates for the paper's qualitative-analysis figure; Findings 1+2 together make the strongest argument for the paper's central thesis that answer-level metrics are necessary but not sufficient for VLM explainability).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day09_robustness}` after `\include{day08_efficiency}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; all 6 robustness figures render (3 before/after pairs); Diagram E's three-branch layout doesn't overlap (check the PDF visually — adjust `below left=`/`below right=` offsets if the three top-row nodes crowd each other).

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day09_robustness.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 9: robustness and the answer-vs-explanation split (Day 9)"
```

---

## Task 12: Chapter 10 — Bounded Completeness: Is the Top Region Actually Necessary? (Day 10)

**Files:**
- Create: `pilot/report/chapters/day10_bounded_completeness.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day10_bounded_completeness}`)

**Interfaces:**
- Consumes: `\citep{arreche2024exai}` (completeness as one of the six metrics, "bounded" variant explicitly scoped down).
- Produces: nothing new (pure rollup + one bounded extra rerun set, no new pipeline stage).

Rich chapter (full `day10_findings.md`). Target ~5 pages.

**Source material:**
- Definition (per `technical-limitations-take-and-mitigation.md`): *"A sample passes the bounded completeness check if the model provides a usable explanation and if perturbing the top-ranked region or top two regions causes a measurable change in the model's answer, confidence proxy, or grounding output."*
- `completeness.py`'s `classify_sample()`: a pure rollup of Chapter 4's `region_extraction.csv` (`top_region_source`) and Chapter 5's `descriptive_accuracy.csv` (`answer_changed_top1`/`top2`) — **no new model calls** for the headline classification. One extra, explicitly bounded set of fresh calls: 159 top-1-only mask reruns (reusing Chapter 4's already-extracted top-1 box) that additionally capture post-mask `worker_boxes`, specifically to check whether the Chapter 5/9 worker-fallback-rerouting artifact also contaminates completeness verdicts.
- **Why no confidence-proxy threshold**: checked first — among Chapter 5's `answer_changed_top1==False` rows, `abs(confidence_drop_top1)` has essentially the same distribution (median 0.032, 75th pct 0.057) as the full population — confidence_drop carries no discriminative signal at this scale; a threshold would inject noise, not real signal.
- **Headline**: `explanation_weak` 88 (54.0%), `explanation_supported` 71 (43.6%), `no_usable_explanation` 4 (2.5%).
- **Hard subsets, reported separately as the plan specifically asks (not blended into the overall rate)**: `quality_of_info == "poor info"` (84 samples, 51.5% of pilot): 39.3% supported / 58.3% weak / 2.4% no_usable — mildly worse, not dramatically so. Multi-rule-violation (n=7, too small for conclusions): 42.9% / 57.1% / 0.0%.
- **Finding 1 — completeness much weaker for PPE (rule_1) than guardrails (rule_3), same mechanism as Chapter 4's area-based ranking**: by rule — rule_1 (hard hat/vest) 33.3% supported (n=63), rule_2 (harness/lanyard) 46.2% (n=26), rule_3 (guardrail/edge barrier) 55.1% (n=49), rule_4 (excavator proximity) 44.0% (n=25). The top-ranked region is the **worker** box in 48/50 (96%) of `ppe_violation`/rule_1 samples; for rule_3, the **guardrail itself** is top-ranked in 33/50 (66%) of `fall_hazard`/rule_3 samples (worker only 12/50, 24%). Across all 163 samples: top-1 region is the worker → 37.5% supported (n=80); top-1 region is the safety/hazard object → 49.4% supported (n=83). Masking the worker is less likely to flip the answer than masking the actual safety object — a real, fixable limitation of the ranking heuristic (rank by "matches the rule's queried object class" before falling back to area).
- **Finding 2 — the worker-fallback-rerouting artifact also inflates a small slice of `explanation_supported`**: of 159 model-region samples, masking top-1 caused the rerun to lose its only detected worker entirely in 12 cases (7.5%). Of 71 `explanation_supported` verdicts, 3 (4.2%) co-occur with this artifact — a corrected "genuinely-supported" count would be 68/163 (41.7%) rather than the naive 71/163 (43.6%). Traced concretely on image `0000642` (rule_2, baseline "compliant"): the model's "worker" box `(436.2, 232.2, 499.8, 419.6)` and its "harness" box `(438.6, 232.2, 495.0, 419.6)` are almost pixel-identical — Florence-2 returned nearly the same region for both queries. Masking the top-ranked "worker" region therefore also erases the harness evidence in the same stroke; the rerun fails to redetect any worker, triggers the fallback branch, and the answer changes — registering as `explanation_supported` when what actually happened is the same degenerate-fallback mechanism from Chapters 5 and 9, this time triggered by this chapter's own masking procedure.
- **Finding 3 — grid-fallback (`no_usable_explanation`) is rare but concentrated in rule_3**: only 4/163 (2.5%) had no native grounding box at all — `0000253` (fall_hazard, rule_3, rich info), `0000361` (fall_hazard, rule_3, poor info), `0000620` (compliant, rule_2, rich info), `0001229` (fall_hazard, rule_3, poor info). 3 of the 4 are rule_3 — "guardrail"/"edge protection barrier" is a harder open-vocabulary detection target than "hard hat" or "worker," consistent with rule_3's already-lowest grounding hit-rate signal from Chapters 3-4.

**Worked example figures:**
- `\includegraphics{completeness/verify/0000642_before.png}` / `\includegraphics{completeness/verify/0000642_after_mask.png}` — Finding 2's traced case (use `subcaption` side by side).

**Standard vs. non-standard:** Standard = completeness/necessity testing via top-k region ablation, directly per `\citep{arreche2024exai}`'s completeness metric, "bounded" variant explicitly scoped down from full completeness over the entire feature space. Non-standard/contribution = the explicit "hard subsets reported separately, never blended" policy, and quantifying exactly how much the fallback-rerouting artifact inflates the headline number — a rigor practice worth recommending generally for any completeness-style metric built on a model with conditional/fallback logic.

- [ ] **Step 1: Write `day10_bounded_completeness.tex`**

Structure: `\chapter{Bounded Completeness: Is the Top Region Actually Necessary? (Day 10)}`, sections `Motivation` (the bounded definition, quoted), `Implementation` (the rollup + bounded extra reruns, the confidence-proxy-threshold decision), `Results` (headline + hard-subset tables), `Finding 1: The Area-Ranking Mechanism, Again` (the rule-by-rule table), `Finding 2: A Worker-Fallback Artifact Inflates the Headline Number` (with figure pair), `Finding 3: Grid-Fallback Is Rare but Concentrated in rule_3`, `Standard vs. Non-Standard Choices`, `Implications for the Paper` (Findings 1 and 2 together are the strongest evidence for "fix the region-ranking heuristic before scaling up" — present as the paper's top recommended next engineering step, tying directly to Chapter 14's task list item 3).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day10_bounded_completeness}` after `\include{day09_robustness}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; the completeness figure pair renders.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day10_bounded_completeness.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 10: bounded completeness (Day 10)"
```

---

## Task 13: Chapter 11 — Aggregating Six Metrics into a Single Cross-Class Picture (Day 11)

**Files:**
- Create: `pilot/report/chapters/day11_analysis_figures.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day11_analysis_figures}`)

**Interfaces:**
- Consumes: `\citep{arreche2024exai}` (the six-metric framework being rolled up).
- Produces: Diagram F (TikZ, the **final** full-pipeline architecture) — the capstone diagram showing how Diagrams A, B, C, D, E (Chapters 1, 3, 4, 6, 9) compose into the complete system. Referenced again in Chapter 13 as what the reproducibility rerun validates end-to-end.

Rich chapter (full `day11_findings.md`). Target ~5 pages.

**Source material:**
- Pure rollup of Chapters 5-10's already-computed CSVs into one summary table and one chart — **no new model calls, no recomputation**. `results/pilot_metric_summary.csv` (6 rows, one per metric), `results/figures/summary/metric_summary_by_class.png`.
- Tooling note worth including: visual review throughout the pilot was done by reading saved PNGs directly (Claude Code's `Read` tool rendering images), not via a Jupyter notebook — `notebooks/` was never actually used.
- **Headline table** (reproduce exactly): descriptive_accuracy 36.2% overall (compliant 28.0%, ppe_violation 28.0%, fall_hazard 46.0%, struck_by_risk 61.5%); visual_sparsity_top5 0.108 overall (0.099 / 0.127 / 0.109 / 0.067); stability 77.5% overall (74.7% / 76.0% / 78.7% / 89.7%); robustness Level 1 80.7% overall (79.5% / 86.0% / 75.0% / 86.5%); bounded_completeness 43.6% overall (32.0% / 36.0% / 58.0% / 61.5%); efficiency ~0.50 GPU-hours @ n=1,000 (aggregate, not per-class). Repeat the `struck_by_risk` caveat: n=13, directional not stable.
- **Finding 1 — descriptive_accuracy and bounded_completeness move together by class, but that's expected, not independent confirmation**: both weakest for compliant/ppe_violation, strongest for fall_hazard/struck_by_risk — but completeness is *built from* descriptive accuracy's own columns by construction, so the shared shape is close to a methodological identity, not a second confirmation. The real, independent explanation (area-ranking promotes worker over PPE item) was already traced in Chapter 10 Finding 1.
- **Finding 2 — visual sparsity runs opposite from descriptive_accuracy/completeness for struck_by_risk**: highest accuracy/completeness (61.5%/61.5%) but lowest sparsity (0.067) — "focused attention" and "decision-critical region" are not the same property. Mechanism: rule_4's attributed objects (excavator, worker) are large structural regions, naturally spreading attention over a bigger area, even though masking that whole region still reliably destroys the spatial relationship the proximity check depends on.
- **Finding 3 — robustness is the flattest metric across classes, and fall_hazard is its worst case (not its best, unlike elsewhere)**: robustness Level 1 varies only 75.0%-86.5% across classes (vs. 30-60 point spans for the other four metrics); fall_hazard is the single worst class here even though it's second-best for descriptive_accuracy/completeness. Matches Chapter 9: fall_hazard pools rule_2 and rule_3, which destabilize for **two different reasons** (rule_2's highest worker_lost rate under occlusion, 19.2%; rule_3's highest grid-fallback rate) — averaging two different failure modes into one class number masks both underlying causes.
- **Finding 4 — efficiency comfortably scales**: summing Chapter 8's extrapolated-to-1,000 hours across all six pipeline steps gives ~0.50 GPU-hours, well under an hour on a single RTX 3070 — the one metric where bigger is unambiguously better news for a future full-dataset study.

**Worked example figure:**
- `\includegraphics{summary/metric_summary_by_class.png}` — the headline rollup chart, the single most important figure in the whole report; give it its own full-width figure environment.

**Standard vs. non-standard:** Standard = cross-metric rollup/aggregation into one summary table+chart, typical of any multi-metric evaluation framework paper, mirroring `\citep{arreche2024exai}`'s own reporting style. Non-standard/contribution = explicitly flagging the descriptive_accuracy/completeness non-independence (avoiding double-counting one finding as two) and the sparsity-vs-completeness divergence as evidence the six metrics measure genuinely different properties, not redundant proxies for one "explanation quality" score.

**Diagram F — TikZ code (final full-pipeline architecture; caption "The complete 11-script pipeline after Day 11 — the composition of Diagrams A-E."):**

```latex
\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=0.9cm and 1.3cm]
\node[block] (s01) {01 env check};
\node[block, right=of s01] (s02) {02 sample\\selection};
\node[block, right=of s02] (s03) {03 baseline\\inference};
\node[block, right=of s03] (s04) {04 region\\extraction};
\node[block, below=1.4cm of s04, xshift=-2.4cm] (s05) {05 descr.\\accuracy};
\node[block, right=of s05] (s06) {06 visual\\sparsity};
\node[block, right=of s06] (s07) {07 stability};
\node[block, right=of s07] (s08) {08 efficiency};
\node[block, below=1.4cm of s06, xshift=1.3cm] (s09) {09 robustness};
\node[block, below=1.4cm of s09] (s10) {10 bounded\\completeness};
\node[block, below=of s10, fill=rulegray] (s11) {11 figures /\\summary};
\draw[arrow] (s01) -- (s02); \draw[arrow] (s02) -- (s03); \draw[arrow] (s03) -- (s04);
\draw[arrow] (s04) -- (s05); \draw[arrow] (s04) -- (s06); \draw[arrow] (s04) -- (s07);
\draw[arrow] (s04) -- (s08); \draw[arrow] (s04) -- (s09);
\draw[arrow] (s05) -- (s10); \draw[arrow] (s06) -- (s10); \draw[arrow] (s07) -- (s10);
\draw[arrow] (s08) -- (s10); \draw[arrow] (s09) -- (s10);
\draw[arrow] (s10) -- (s11);
\end{tikzpicture}
\caption{The complete 11-script pipeline after Day 11 — the composition of Diagrams A-E.}
\label{fig:diagram-f}
\end{figure}
```

- [ ] **Step 1: Write `day11_analysis_figures.tex`**

Structure: `\chapter{Aggregating Six Metrics into a Single Cross-Class Picture (Day 11)}`, sections `Motivation`, `Architecture: The Complete Pipeline` (Diagram F, explicitly note it composes Diagrams A/B/C/D/E from Chapters 1/3/4/6/9), `Results` (the headline table), `Worked Example` (the summary chart figure), `Finding 1` through `Finding 4`, `Standard vs. Non-Standard Choices`, `Implications for the Paper` (this chapter's table is essentially the paper's Results-section centerpiece).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day11_analysis_figures}` after `\include{day10_bounded_completeness}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; Diagram F's 11-node layout doesn't overlap (check the PDF visually — this is the most complex diagram in the report; adjust `xshift`/`below=`/`right=of` offsets if nodes crowd, e.g. widen `node distance` or split into two rows with more vertical spacing); the summary chart figure renders full-width.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day11_analysis_figures.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 11: six-metric rollup and final pipeline diagram (Day 11)"
```

---

## Task 14: Chapter 12 — Synthesizing Findings into a Pilot Report (Day 12)

**Files:**
- Create: `pilot/report/chapters/day12_pilot_report.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day12_pilot_report}`)

**Interfaces:**
- Produces: nothing new (writing activity, not a code/pipeline change — no diagram).

Thin chapter (no dedicated `findings.md`; source is the report artifact itself + commit message). Target ~2-3 pages.

**Source material:**
- Goal per `two-week-pilot-plan.md` Day 12: write a 4-6 page technical memo synthesizing Days 1-11 — motivation, dataset, model, methods, six metrics, preliminary findings, limitations, next steps.
- Artifact: `pilot_report_draft.md`, commit `c2a1f85`.
- **Editorial policy adopted, worth reporting explicitly as a methodological choice**: quote `pilot_report_draft.md`'s own header verbatim — *"Every number below traces to a specific CSV or figure produced in Days 1-11 (`pilot/results/`); none are invented or estimated for this writeup."*
- Structure chosen for the report (which this very chapter set later expanded): Motivation → Dataset → Model and the grounding-proxy adaptation → Methods summary (one table mapping each metric to its method) → Results (headline table) → Cross-cutting findings (synthesizing discoveries that recur across multiple days — e.g. the fallback-rerouting mechanism appearing independently in Chapters 5, 9, and 10; the area-ranking confound appearing independently in Chapters 4, 6, 7, and 10) → Limitations (explicit, not papered over) → Next steps / where Professor Abdallah's guidance is wanted.
- State explicitly: **noticing the same mechanism recur across independently-run days is a distinct, valuable activity from any single day's individual finding** — this synthesis work is what turns 11 days of isolated results into a coherent picture.
- Standard vs. non-standard: standard = a structured technical report synthesizing a multi-stage empirical pilot. Non-standard/notable = the strict "every number must trace to a specific file" sourcing discipline, maintained even under a 14-day timebox.

- [ ] **Step 1: Write `day12_pilot_report.tex`**

Structure: `\chapter{Synthesizing Findings into a Pilot Report (Day 12)}`, sections `Motivation`, `The Report's Structure and Editorial Policy` (quote the "nothing invented" line), `Cross-Cutting Synthesis as Its Own Activity` (the two named recurring-mechanism examples), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (this is the natural place for a short "process" paragraph describing how findings were tracked day-by-day and only synthesized after, not retrofitted). No figures, no diagram, no new citations.

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day12_pilot_report}` after `\include{day11_analysis_figures}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day12_pilot_report.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 12: synthesizing the pilot report (Day 12)"
```

---

## Task 15: Chapter 13 — From-Scratch Reproducibility and a Real Bug Found by Testing the Process (Day 13)

**Files:**
- Create: `pilot/report/chapters/day13_reproducibility.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day13_reproducibility}`)

**Interfaces:**
- Produces: nothing new (verification activity — no architecture change, though it retroactively validates Diagram F from Chapter 11).

Rich chapter (full `day13_findings.md`). Target ~4-5 pages.

**Source material:**
- Per the plan's Day 13 verification criterion: *"the from-scratch 20-sample rerun completing without manual intervention is the actual test."* Exported the exact committed `pilot/` tree at `HEAD` (`git archive HEAD pilot`) into a throwaway directory outside the repo, created a brand-new venv, ran `pip install` from a clean slate, ran `scripts/01` through `scripts/11` in order against a fresh 20-sample subset (`SAMPLES_PER_CLASS` patched from 50 to 5 in that throwaway copy only — 5/class × 4 classes = 20, never touching the real `pilot/data` or `pilot/results`).
- **Result: full pipeline reproduces from scratch, exit 0 on every script.** `pytest tests/ -v`: 61/61 passed, same as the main venv. `01_check_environment.py`: GPU confirmed (`cuda:0`, fp16), gated dataset access worked using the existing user-level HF token (no extra login — `huggingface_hub`'s token cache lives at `~/.cache/huggingface`, not inside the venv). `02`: wrote exactly 20 rows, 5/class, no underfilled-class warning. `03`-`11`: all exit 0, all expected row counts (`04`: 20/20, grid-fallback 5.0%, 1/20; `08`: 20/20 + 15-sample calibration; `09`: 20/20 + patch-stretch; `11`: wrote summary CSV + chart).
- **No manual intervention was needed at any step** — the install order (pinned-CUDA `torch` first, then `requirements.txt`, then `pip install -e .`) worked exactly as documented; the exact-version pin (`torch==2.12.1`) meant the second install step didn't silently replace the CUDA build with a CPU one (verified directly: `torch.__version__` was `2.12.1+cu130` and `torch.cuda.is_available()` was `True` both before and after `pip install -r requirements.txt`).
- **Two real things this rerun surfaced — make this the chapter's centerpiece narrative:**
  1. **A genuine bug in `11_make_figures.py`, found and fixed.** The chart's legend and title hardcoded `"n=50"` / `"n=13"` and `"163-sample pilot"` — correct for the main pilot, silently wrong for any other sample size (this 20-sample run would have shown a chart mislabeled "163-sample pilot" with "n=50" legends next to bars actually computed from 5 samples each). Fixed to read `primary_class` value counts and `len(da)` directly from the already-loaded `descriptive_accuracy.csv` instead of hardcoding either number. Re-ran `11_make_figures.py` against the real 163-sample data after the fix — output byte-identical in its numbers (the fix only touches chart labels, no computation), chart visually re-confirmed correct.
  2. **A transient network error during `05_descriptive_accuracy.py`'s dataset streaming, auto-recovered.** `IncompleteRead`/`ProtocolError` while streaming the test-split parquet from the Hub, retried automatically by `huggingface_hub`'s built-in retry logic (3 retries, exponential backoff) and succeeded. Not a code bug — flagged as a real source of pipeline flakiness worth being aware of for future full-dataset runs.
- **20-sample numbers vs. the real 163-sample pilot — directionally consistent, as expected**: descriptive_accuracy 36.2% (163) vs. 30.0% (20); visual_sparsity_top5 0.108 vs. 0.092; stability_answer_agreement 77.5% vs. 93.3% (notably higher — expected sampling noise at n=20, one or two samples landing differently swings the mean substantially at this size, not a discrepancy that calls the pipeline's correctness into question); robustness_level1 80.7% vs. 83.8%; bounded_completeness 43.6% vs. 35.0%; efficiency @ n=1,000 ~0.50 vs. ~0.54 GPU-hours. Four of six land close; the rule-level ordering Chapters 6/7/10 already explained mechanistically (excavator-class objects score lowest on sparsity, rule_4 highest on stability/robustness) holds in both runs — the more meaningful reproducibility signal than exact percentage match.
- Cleanup: the throwaway venv + data + results were deleted after the check — scratch space only, not a tracked artifact.

**Standard vs. non-standard:** Standard = a clean-room reproducibility check (fresh venv, fresh clone, documented install steps) — increasingly expected practice for empirical ML papers' artifact-evaluation/reproducibility checklists (e.g. NeurIPS/ACL). Non-standard/contribution = explicitly treating the reproducibility check itself as a **test of the pipeline's robustness** (it caught a real bug), not merely a formality — and deliberately choosing a reduced n (20 vs. 163) specifically to keep the check fast while still exercising every code path.

- [ ] **Step 1: Write `day13_reproducibility.tex`**

Structure: `\chapter{From-Scratch Reproducibility -- and a Real Bug Found by Testing the Process (Day 13)}`, sections `Motivation` (the verification criterion, quoted), `Method` (the clean-room procedure), `Result: Full Pipeline Reproduces, Exit 0 on Every Script` (the step-by-step table), `Two Real Things This Rerun Surfaced` (both numbered findings, with the bug-fix narrative as the highlight), `20-Sample vs. 163-Sample Numbers` (the comparison table, framed as "reproducing the pipeline, not the exact numbers"), `Standard vs. Non-Standard Choices`, `Implications for the Paper` (this chapter doubles as the paper's Reproducibility section — frame the bug-found-by-testing-the-process narrative as a concrete argument for why reproducibility checks belong in the pilot phase, not deferred to camera-ready).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day13_reproducibility}` after `\include{day12_pilot_report}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day13_reproducibility.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 13: from-scratch reproducibility and the figure-label bug (Day 13)"
```

---

## Task 16: Chapter 14 — Packaging the Pilot for a Research Collaboration Meeting (Day 14)

**Files:**
- Create: `pilot/report/chapters/day14_meeting_package.tex`
- Modify: `pilot/report/chapters/main.tex` (add `\include{day14_meeting_package}`)

**Interfaces:**
- Produces: nothing new (packaging/communication activity — the report's closing chapter; cross-references every prior chapter by number).

Thin, closing chapter. Target ~3 pages.

**Source material:**
- Goal per `two-week-pilot-plan.md` Day 14: finalize one-page pitch, pilot report, sample visualizations, task plan for the full study; prepare a 5-minute explanation (what was tested, what worked, what failed, where Prof. Abdallah's expertise is needed).
- Artifacts: `report/one_pager_summary.md`, `report/talking_points.md`, `report/key_visualizations.md` (curated subset of 60+ saved figures, each tied to a specific finding, "not picked for being pretty"), `report/full_study_task_list.md` (each item framed as a decision needing Prof. Abdallah's judgment, not a generic to-do).
- One-pager headline: does Abdallah's six-metric framework (built for tabular intrusion-detection DNNs) transfer to a VLM making visual judgments? Tested on construction-safety VQA, Florence-2-base-ft (231M params, single RTX 3070), 163 images. Headline answer: **yes, with two real, quantified adaptations** (the grounding-proxy reframing of Chapter 3; the person-relative redesign, 9× sensitivity gain), not zero changes.
- **Curated key visualizations (6 picks, each tied to a specific finding — cross-reference the chapter, don't re-include the images)**: (1) `summary/metric_summary_by_class.png` — Chapter 11's headline rollup; (2) `baseline/0000007_rule_1.png` — Chapter 3's grounding-proxy adaptation working; (3) `descriptive_accuracy/0000037_rule_1_flip.png` — Chapter 5's descriptive accuracy on a real sample; (4) `sparsity/0000023_rule_1_hard_hat.png` + `sparsity/0000079_rule_3_guardrail.png` — Chapter 6's focused-vs-scattered pair; (5) `robustness/verify/0000034_before.png`/`_after_occlude.png` — Chapter 9's stable-answer-hiding-a-drifted-explanation case; (6) `efficiency/per_sample_cost.png` — Chapter 8's compute-is-not-the-obstacle figure.
- **Full study task list — the six decisions framed for Prof. Abdallah, each cross-referenced to the chapter that produced its evidence**: (1) attribution method — is decoder→encoder cross-attention an acceptable LRP/DeepLIFT stand-in, or does a transformer-native method need to be added? (Chapter 6). (2) statistical rigor at small n, especially `struck_by_risk` (n=13) (Chapters 2, 11). (3) region-ranking heuristic — rank by "matches the rule's queried object class" before falling back to area (Chapters 4, 10). (4) full-dataset scaling (3,004 images) vs. fixing the `struck_by_risk` class-imbalance — compute confirmed not the constraint either way (Chapter 8). (5) a real Level-3 robustness study — the current 20-image/n=4-meaningful patch test is a plausibility check, not a result (Chapter 9). (6) revisit rule_2's two prompt phrasings, which may ground different physical objects, not synonyms (Chapter 9, Finding 3).
- Standard vs. non-standard: standard = preparing a stakeholder-facing summary package (one-pager + talking points + curated visuals) at the close of a feasibility study. Non-standard/notable = framing every open item as an explicit decision needing the collaborator's judgment, with pilot evidence cited per item, rather than a generic "future work" list.

- [ ] **Step 1: Write `day14_meeting_package.tex`**

Structure: `\chapter{Packaging the Pilot for a Research Collaboration Meeting (Day 14)}`, sections `Motivation`, `Artifacts Produced`, `The Headline Answer` (state the yes-with-two-adaptations answer plainly, closing the loop opened in Chapter 1), `Curated Visualizations` (the 6-item cross-reference list — cross-reference by `\nameref`/chapter number, do not re-embed the images a third time), `Six Decisions for Professor Abdallah` (the cross-referenced task list), `Standard vs. Non-Standard Choices`, `Closing Synthesis` (connect back to Chapter 1's opening question — does the six-metric framework transfer? — with the headline answer), `Implications for the Paper` (this chapter is the natural home for the paper's Conclusion and Future Work section; the six decisions map almost directly onto candidate future-work bullets).

- [ ] **Step 2: Add to `main.tex`**

Insert `\include{day14_meeting_package}` after `\include{day13_reproducibility}` — this is the last `\include` line, immediately before `\bibliographystyle{plainnat}`.

- [ ] **Step 3: Compile and verify**

Run: `cd "pilot/report/chapters" && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; all 14 chapters now appear in the table of contents in order.

- [ ] **Step 4: Commit**

```bash
git add pilot/report/chapters/day14_meeting_package.tex pilot/report/chapters/main.tex
git commit -m "Add Chapter 14: meeting package and closing synthesis (Day 14)"
```

---

## Task 17: Final full-report verification pass

**Files:** none created; verification only.

- [ ] **Step 1: Clean rebuild from scratch**

Run: `cd "pilot/report/chapters" && latexmk -C && latexmk -pdf -interaction=nonstopmode main.tex`
Expected: exit code 0; `main.pdf` exists; page count is in the expected ~55-75 page range given 9 rich chapters (~4-7 pages each) + 5 thin chapters (~2-3 pages each) + front matter + bibliography.

- [ ] **Step 2: Check the log for any remaining warnings**

Run: `grep -iE "undefined|multiply.defined|overfull \\\\hbox.*[5-9][0-9][0-9]\\.[0-9]pt|! " pilot/report/chapters/main.log`
Expected: no `Reference ... undefined`, no `Citation ... undefined`, no `! LaTeX Error` lines. Overfull-hbox warnings under ~50pt are cosmetically fine and can be ignored; anything flagged above is worth a quick look (usually a TikZ node label too wide for its box, or a table column too narrow — fix inline if found).

- [ ] **Step 3: Spot-check all 6 architecture diagrams render without node overlap**

Open `main.pdf` (or convert relevant pages) and visually confirm Diagrams A (Ch.1), B (Ch.3), C (Ch.4), D (Ch.6), E (Ch.9), F (Ch.11) each render with no overlapping boxes/labels and arrows pointing the right direction. Diagrams D, E, and F are the most node-dense (4-11 nodes) and most likely to need a `node distance` or `xshift`/`yshift` tweak — fix any found directly in that chapter's `.tex` file and recompile.

- [ ] **Step 4: Confirm every `\includegraphics` resolved**

Run: `grep -i "file.*not found\|could not locate" pilot/report/chapters/main.log`
Expected: no matches. If any figure path is wrong, cross-check it against the verified figure listing in this plan's chapter tasks (Tasks 5, 6, 7, 8, 9, 11, 12 — every figure path used was confirmed to exist on disk during planning).

- [ ] **Step 5: Final commit**

```bash
git add pilot/report/chapters/
git commit -m "Final verification pass: clean rebuild of all 14 chapters"
```

---

## Self-Review

**Spec coverage:** All 14 chapters from the design doc's table are covered (Tasks 3-16, one each). Setup (Task 1), bibliography (Task 2), and final verification (Task 17) cover the remaining design-doc sections (file layout, bibliography, verification plan). The 6 architecture diagrams (A-F) match the design doc's exact turning-point list (Chapters 1, 3, 4, 6, 9, 11) — no more, no fewer. The "scale to evidence" depth rule is reflected in explicit page-count targets per task (thin: Tasks 4, 6, 14, 16 at 2-3 pages; rich: Tasks 5, 7, 8, 9, 10, 11, 12, 13, 15 at 4-7 pages).

**Placeholder scan:** No "TBD"/"TODO" in any task. Every chapter task's "Source material" section contains real, pre-verified numbers and quotes pulled directly from the actual `dayN_findings.md` files, source code, or result CSVs — not generic instructions to "summarize findings." The only intentionally-deferred content is chapter *prose* (the connecting sentences), which is explicitly named as the task's actual work in the "Adaptation note" at the top of this plan, not a hidden gap.

**Type/interface consistency:** Figure paths are consistent with `\graphicspath{{../../results/figures/}}` set once in `preamble.tex` (Task 1) — every later `\includegraphics{...}` call in Tasks 5-13 uses a path relative to that root, matching the verified `find results/figures -type f` listing gathered during planning. Bib keys introduced in Task 2 (`arreche2024exai`, `arreche2025whitebox`, `xiao2024florence2`, `abnar2020rollout`, `kokhlikyan2020captum`, `bach2015lrp`, `shrikumar2017deeplift`, `constructionsite10k`) are the only keys `\citep`/`\citet` in Tasks 3, 5, 6, 7, 8, 9, 10, 11 — no chapter invents a new key. `\include` ordering in `main.tex` is built incrementally and strictly day-ordered across Tasks 3-16 (each task's Step 2 inserts immediately after the previous task's chapter).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-23-pilot-daily-chapters-plan.md`.

**Recommended: Inline Execution (`superpowers:executing-plans`), not Subagent-Driven.** Rationale specific to this plan: this is a 14-chapter narrative document, not 14 independent code modules. A fresh subagent per chapter would need to re-derive the shared voice, the cross-chapter forward/backward references ("flagged forward to Chapter 9," "the mechanism Chapter 4 introduced"), and the running themes (the fallback-rerouting artifact spanning Chapters 5/9/10; the area-ranking confound spanning Chapters 4/6/7/10) — all of which are far cheaper to carry in one continuous session's context than to re-establish per subagent. Execute tasks 1-17 in order, in this session, compiling and committing after each one.
