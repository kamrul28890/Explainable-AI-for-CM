# Handoff prompt — continue MEETING_BRIEFING.md

Paste everything below this line to Codex as the task prompt.

---

## Context

I'm preparing a briefing document for a meeting with my research advisor, Professor Mustafa Abdallah, about an XAI (explainable AI) research project. The project tests whether a six-metric XAI evaluation methodology — built and validated on tabular network-intrusion-detection models — transfers to a Vision-Language Model (Florence-2) doing construction-site safety judgments. A two-week pilot (163 images) is complete; a scale-up is in progress (Phases 1–3 of a 6-phase plan are done or in progress).

I've been writing a self-contained Markdown briefing document with Claude, section by section, with the professor's meeting today. I'm out of budget partway through. You're picking up where I left off.

**The file:** `professor-package/MEETING_BRIEFING.md` (already has §1–§5 written and reviewed by me). Read it in full before doing anything else — it establishes the voice, structure, and level of detail you must match exactly.

## The document's house style — follow this precisely

- **Plain language, short sentences.** Technical terms are fine, but if a concept could be confusing, follow it immediately with a concrete example. (E.g., §1 explains "tabular" data with a network-intrusion-detection table example before using the word freely.)
- **No unearned takeaways.** Every subsection explains, in order: (a) briefly what we're doing here, (b) the technicalities / what went into it, (c) what came out. For pipeline-stage sections specifically: was it okay? If not, why, what we changed, and did the outcome actually improve?
- **Every number must trace to a real file in the repo.** Never invent a statistic. If you state a number, you must have grepped/read it from the actual code, results CSVs, or the existing LaTeX report chapters (`pilot/report/chapters/day*.tex`, `pilot/report/scaleup/phase*.tex`, `professor-package/project_report.tex`). If you're not sure a number is real, say so and flag it rather than guessing.
- **Frame the six metrics as an established methodology being applied to a new domain** — never as "Professor Abdallah's framework" or implying we're "taking" something from him. This was an explicit correction from the user; see the second-to-last paragraph pattern already in §1 ("Do we change the metrics, or adopt them as-is?").
- **Figures are marked as placeholders, not embedded.** The user wants a Markdown draft first; a PDF with actual images comes later. Wherever an image would help, add a blockquote note in this exact style (see §5 for a dozen live examples):
  > **[Figure — for the final PDF]** description of what to show + exact repo file path(s).
  Do not try to embed actual images in the Markdown.
- **Tables** are used for numeric results and comparisons; Mermaid diagrams for architecture/flow (see §3 for the pattern).

## What's already done (§1–§5) — do not rewrite, only read for style/content continuity

1. **§1 Research Objective & Problem Statement** — the tabular-vs-VLM contrast, why this matters, the research question.
2. **§2 Proposed Methodology** — the grounding-proxy strategy (Florence-2 can't do VQA, so we detect objects + apply a geometric rule), and how each of the six metrics is operationalized on images.
3. **§3 Implementation Plan & Architecture** — the 11-stage pipeline flowchart (Mermaid) + a layered architecture diagram (config → library → scripts → outputs).
4. **§4 Components, One by One** — the dataset (ConstructionSite, 10,013 images, local Parquet cache at `pilot/data/constructionsite/`), the model (Florence-2-base-ft, 231M params), and the supporting library modules (`prompts.py`, `regions.py`, `attribution.py`, `perturbations.py`, `metrics/`). Includes a table bridging the four safety classes to exactly how each is turned into Florence-2 grounding queries.
5. **§5 Pipeline Transformation, Stage by Stage** — the big one. Walks all 11 pipeline stages (environment check → sample selection → baseline inference → region standardization → the six metrics → roll-up + reproducibility check), each with what-happened/was-it-okay/what-we-changed/did-it-help, real numbers, and figure placeholders. Ends with 4 cross-cutting findings tying the stages together.

## What's left to do — in order

### Immediate: polish pass on §5 (quick, do this first)
The user asked several follow-up questions that I answered by editing §4 and §5 directly (dataset storage path, the four-classes-to-Florence bridge table, the struck_by_risk full-dataset numbers, a compliant round-robin example, and figure placeholders for stages 3/4/5/6/7/9/10). **Verify these edits read cleanly and consistently** — check for any rough transitions where I inserted new paragraphs into existing prose. Fix anything awkward. Also: stage 06 and stage 09's figure notes were added but double check every other stage (01, 02, 08, 11) — decide if they also warrant a figure placeholder note (01 and 08 probably don't need one; 02 might benefit from a simple class-count bar chart note; 11 might reference `pilot/results/figures/summary/metric_summary_by_class.png` or `professor-package/figures/summary.png`).

### §6 — The Full Plan & Where We Are Now (Part B: the scale-up)
This section should:
1. **State the central goal**: make the metrics measure **model faithfulness**, not the geometric proxy's sensitivity. (This is the theme §5 ends on — pick it up directly.)
2. **Walk the phased plan** (source: `professor-package/SCALEUP_IMPLEMENTATION_PLAN.md`, sections 2–7): Phase 0 (four decisions needing sign-off — save the actual decision content for §7, just name them here), Phase 1 (foundational fixes), Phase 2 (metric-validity fixes), Phase 3 (reproducibility/stats), Phase 4 (attribution streams), Phase 5 (model/dataset scope — Qwen2.5-VL, SODA, CMA).
3. **Report actual progress**, stage by stage, in the same "what we did / what came out" style as §5:
   - **Phase 1 (DONE, commit `68c90da`)** — 7 items. Source: `pilot/report/scaleup/phase1_1_rule_aware_ranking.tex` through `phase1_7_hardcoded_audit.tex`. Read each chapter's Results/Implications section for before/after numbers (e.g., Phase 1.1's rule-aware ranking narrowing the "mask worker vs mask object" gap; Phase 1.2's multi-label recovery from 13→24 struck_by_risk in the test split, already referenced in §5 Stage 02 — don't just repeat it, build on it here with the full phase writeup).
   - **Phase 2 (DONE, commit `ac69a27`)** — 6 items. Source: `pilot/report/scaleup/phase2_1_descriptive_validity.tex` through `phase2_6_efficiency.tex`.
   - **Phase 3 (IN PROGRESS, commit `84ec951` + uncommitted work)** — Phase 3.1 (stats hardening: bootstrap CI, Wilcoxon, min-n) is committed. Phase 3.2 (multi-seed) is **written but uncommitted** — read `pilot/report/scaleup/phase3_2_multiseed.tex` (already read in this session; key numbers: 5 seeds {42-46}, per-class counts identical every seed, mean pairwise Jaccard overlap 0.222, 475 distinct images across 5 seeds vs 200 per seed, struck_by_risk shows zero seed variation because all 13 are selected every time — "exhaustion, not stability"). **Check `git status` and `git diff` to see exactly what's uncommitted** (`pilot/scripts/02_select_samples.py`, `pilot/src/xai_pilot/config.py`, `pilot/src/xai_pilot/stats.py`, `pilot/tests/test_stats.py`, `pilot/report/scaleup/main.tex` were modified; `phase3_2_multiseed.tex` is a new untracked file) — describe this as genuinely in-progress work, not finished.
   - **Phases 4–5 — NOT STARTED**, gated on Phase 0 decisions. Just note this here; the decisions themselves are §7's content.
4. Keep the "what came out" numbers real — pull them from the phase chapters, don't approximate.

### §7 — Open Confusions & Proposed Solutions
This is the "ask the professor" section. Content:
1. **The four Phase-0 decisions** from `SCALEUP_IMPLEMENTATION_PLAN.md` §2 (D1 attribution method, D2 model scope, D3 multi-label class labeling, D4 detection task token) — for each: state the confusion/fork plainly, the recommended default (already in the plan document), and why it's not a slam-dunk default (what changes if he picks differently). Use the plan's own "Recommended default" and "Consequence if changed" columns as your source, but write them in the document's plain-language style, not copied verbatim.
2. **The minimum-n / statistics standard** — also flagged as needing sign-off in the plan (§2, the line after the D1-D4 table).
3. Check `feedbacks-for-full-project.md` and `my-notes.md` (both already modified/read earlier this session) for any additional genuine open questions or confusions the user raised in earlier self-review that haven't been resolved by Phase 1-3 work yet — cross-reference against what's already fixed before including something, since some feedback in those files may already be addressed by completed phases.
4. End with a short, clearly-marked list of **decisions requested of Professor Abdallah** — a scannable summary for the actual meeting (the existing `professor-package/project_report.tex` has a "Six decisions requested" list near its end you can use as a starting reference, but reconcile it against what Phase 1-3 has already resolved so you don't ask about something already settled).

### After §7 is drafted
- Do a full read-through of the entire document for consistency (terminology, tense, the "what we're doing / details / what came out" rhythm, cross-references between sections like "see §6" actually pointing to real content).
- Do NOT convert to PDF yet — the user explicitly wants the Markdown finalized and reviewed first.
- Leave `professor-package/CODEX_HANDOFF.md` (this file) in place until the user confirms it's no longer needed — don't delete it yourself.

## Working style notes (from this session)

- The user reviews **section by section** and says "next" or "go" to proceed, or gives specific corrections (like the "don't say 'his framework'" correction applied to §1, or the batch of 10 follow-up questions applied to §4/§5). Don't write §6 and §7 both in one uninterrupted pass without checking in — post §6, summarize what you wrote and wait, the same way this conversation did after each section.
- The user is a student; the professor is Mustafa Abdallah. Tone is professional but plain — this is a working document, not a polished publication.
- Don't use emojis. Don't add unrequested sections or reorganize what's already approved.
- If you find a number in the pilot/scale-up chapters that contradicts something already written in §1–§5, flag it to the user rather than silently "fixing" §1–§5 — those sections were already reviewed and approved.
