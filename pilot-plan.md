# Construction-VLM XAI Pilot — Implementation Plan

> **For agentic workers:** Execute task-by-task with checkpoints. Day 1-4 tasks include exact code; Day 5-14 tasks specify exact deliverables/interfaces but final parameter tuning happens against real outputs from Day 1-4 (this is a research pilot, not a fully pre-determined feature — see "Pacing" note below).

## Context

`workflow.md` and `two-week-pilot-plan.md` (plus `Research-prposal.md` and `technical-limitations-take-and-mitigation.md` in this folder) lay out a two-week feasibility pilot: prove that Professor Mustafa Abdallah's six-metric XAI evaluation framework (Descriptive Accuracy, Sparsity, Stability, Efficiency, Robustness, Completeness — originally built for tabular network-intrusion DNNs) can be adapted to a small construction-safety Vision-Language Model setting, ahead of a collaboration pitch meeting.

Nothing has been built yet — the folder currently contains only proposal/analysis documents and PDFs, no code. This plan turns the two source documents into one concrete, executable pipeline, resolving the places where they were aspirational rather than technically verified (dataset access, Florence-2's actual prompting capabilities, Windows-specific install issues, GPU availability).

**Decisions already confirmed with the user:** new dedicated git repo in this folder; hybrid code style (tested Python modules + thin notebooks for visual review); start "Day 1" today (2026-06-22) with a flexible end date; scope limited to the technical pilot only (outreach email / 1-pager pitch is a separate future conversation).

## Verified Ground Truth (resolves Day-1 unknowns before they cost a day)

I checked these directly rather than assuming the docs were current:

| Item | Finding |
|---|---|
| Dataset identity | "ConstructionSite 10k" = [`LouisChen15/ConstructionSite`](https://huggingface.co/datasets/LouisChen15/ConstructionSite) on HF. 10,013 images (7,009 train / 3,004 test), gated, `cc-by-nc-4.0`. Companion code: [`LouisChen15/ConstructionSite-10k-Implementation`](https://github.com/LouisChen15/ConstructionSite-10k-Implementation) (Gemini-only inference scripts, but has reusable eval-scoring scripts under `Evaluations/safety_violation_vqa` and `Evaluations/object_detection` worth skimming for IoU/answer-matching conventions before writing ours). |
| Gated access | **Already granted.** The cached HF token for account `kamrul28890` can already call `dataset_info`/list files on the repo — confirmed via API. No waiting on approval. |
| Exact schema | Columns: `image, image_id, image_caption, illumination, camera_distance, view, quality_of_info, rule_1_violation..rule_4_violation` (each `{bounding_box: [[x0,y0,x1,y1],...] (normalized 0-1), reason: str}` or `null` if compliant), `excavator`, `rebar`, `worker_with_white_hard_hat` (grounding boxes). |
| Rule → hazard-class mapping (from the dataset card, not guessed) | Rule 1 = basic PPE (hard hat/vest/clothing/eye protection) → **ppe_violation**. Rule 2 = harness use at height ≥3m without edge protection → **fall_hazard**. Rule 3 = edge protection/guardrails at height or excavation ≥3m → **fall_hazard**. Rule 4 = worker in excavator blind spot/operating radius → **struck_by_risk**. All four `null` → **compliant**. |
| Florence-2-base-ft | MIT license, ungated, 231.6M params, on HF as `microsoft/Florence-2-base-ft`. |
| **Florence-2 has no native free-form VQA task token.** | Its real task tokens are `<CAPTION>` / `<DETAILED_CAPTION>` / `<MORE_DETAILED_CAPTION>`, `<OD>`, `<DENSE_REGION_CAPTION>`, `<REGION_PROPOSAL>`, `<CAPTION_TO_PHRASE_GROUNDING>`, `<OPEN_VOCABULARY_DETECTION>`, `<REFERRING_EXPRESSION_SEGMENTATION>`, `<OCR>`/`<OCR_WITH_REGION>`. There is no `<VQA>` token in the base/ft checkpoints. The "ask it `Is the worker wearing a hardhat?`" framing in `workflow.md`/`Research-prposal.md` will not work as literally described — see Design Decision #1 below for the fix. This is the same class of "assumed-transferable, actually isn't" gap that `technical-limitations-take-and-mitigation.md` already flagged for LRP/iNNvestigate; it just goes one level deeper (the *model*, not only the XAI tooling). |
| Local GPU | RTX 3070, 8GB VRAM, driver supports CUDA 13.1 — plenty for a 231M-param model with Captum gradients at batch size 1. |
| Local Torch install | **Currently `torch==2.11.0+cpu`** despite the GPU being present — must reinstall the CUDA build on Day 1 or everything runs on CPU. |
| Already installed (global Python 3.10.11, no venv yet) | `transformers 5.2.0`, `datasets 4.5.0`, `accelerate`, `huggingface_hub`, `opencv-python`, `Pillow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scikit-image`, `scipy`, `einops`. **Missing:** `captum`. |
| Disk | D: drive has 76GB free — dataset parquet (~4.5GB) + model weights (~460MB) fit easily. |

## Design Decisions (the real adaptation work)

1. **VQA proxy via grounding, not free-text Q&A.** Each safety rule becomes a grounding query instead of a yes/no question to a model that can't answer one. For rules 1-3 (presence-based: hard hat, harness, guardrail) we call `<OPEN_VOCABULARY_DETECTION>` with the safety-object phrase; if Florence-2 returns a box, the proxy answer is "compliant", otherwise "violation". For rule 4 (proximity-based) we detect `worker` and `excavator` independently and compute a spatial-overlap/distance check between their boxes; overlap above a threshold = "hazard". This keeps the *explanation unit* exactly what the proposal wants (a visual region tied to a safety concept) while using capabilities Florence-2 actually has. The model's own native grounding box becomes one explanation source; attention-rollout/Captum attribution (Step 3 of `workflow.md`) becomes the second, finer-grained source. Confidence proxy = mean token-level generation probability from `model.generate(..., output_scores=True, return_dict_in_generate=True)`, exactly the "confidence proxy" language the source docs already use (they hedge on this, so this isn't a contradiction).
2. **Stability test needs real stochasticity.** Florence-2's default decoding (beam search, no sampling) is deterministic — three reruns of the literal pipeline would be bit-identical and "100% stable" by construction, which proves nothing and would look naive in front of Professor Abdallah. Day 7 will run 3x with `do_sample=True, temperature=0.7, num_beams=1` as the actual stability test, and report one deterministic beam-search run separately as a labeled "decoding ceiling" reference. This refinement is mine, not in the source docs, and the report should say so explicitly.
3. **Windows + Florence-2's custom code imports `flash_attn`,** which has no easy Windows wheel. Standard community fix: monkey-patch `transformers.dynamic_module_utils.get_imports` to strip `flash_attn` out before `from_pretrained(..., trust_remote_code=True)`, and force `attn_implementation="sdpa"`. Implemented once in `model.py`, used everywhere.
4. Everything else (bounded completeness to top-1/top-2 regions, robustness Level 1 = visual perturbations + a small Level 2 = prompt rewording, separate visual/text sparsity streams, deferring LRP/DeepLIFT/full adversarial retraining) follows `technical-limitations-take-and-mitigation.md` directly — no changes needed there, it's already the right scope.
5. **`workflow.md` Step 5's mini adversarial-patch test (20 images, synthetic hard-hat sticker)** doesn't have its own day in `two-week-pilot-plan.md`'s Day 9. Folding it into Day 9 as an explicitly bounded, time-boxed stretch task (20 images, can slip without threatening the day's core deliverable) honors both documents without overpromising — consistent with `technical-limitations-take-and-mitigation.md`'s "Level 3 should be a small part, not a full promise."

## Pacing note

Days 1-4 (environment, sampling, baseline inference, region extraction) get full code now because they're fully determined by the verified facts above. Days 5-14 (the six metrics + analysis + report) get concrete formulas, exact file/function targets, and exact deliverables, but not pre-written numeric thresholds where those genuinely depend on what Day 1-4 produces (e.g., how often Florence-2 actually returns a grounding box at all) — those get set from real data, not guessed now. This mirrors `technical-limitations-take-and-mitigation.md`'s own stance: bound the promise to what's technically verifiable.

## Repo & File Structure

New git repo rooted at this folder (`Explainable-AI-Mustafa-Abdallah/`), code under a `pilot/` subdirectory so the existing proposal docs stay at the root:

```
Explainable-AI-Mustafa-Abdallah/
├── (existing docs: workflow.md, two-week-pilot-plan.md, Research-prposal.md, ...)
└── pilot/
    ├── .gitignore                 # data/, results/*.png caches, .venv/
    ├── requirements.txt
    ├── pyproject.toml             # `pip install -e .` for src/xai_pilot
    ├── src/xai_pilot/
    │   ├── config.py              # paths, HF_DATASET_ID, MODEL_ID, RULE_TO_CLASS, seeds
    │   ├── model.py                # load_florence2() with flash_attn patch; run_task(image, task_token, text_input=None)
    │   ├── data.py                 # load_construction_site(), classify_image(), select_balanced_sample()
    │   ├── prompts.py               # RULE_QUERIES, build_grounding_query(rule_id)
    │   ├── inference.py             # answer_rule(image, rule_id) -> AnswerResult(answer, boxes, confidence, timing)
    │   ├── regions.py               # mask_region(), blur_region(), grid_fallback_regions(), iou()
    │   ├── attribution.py           # attention_rollout(), gradient_x_attention() via Captum
    │   ├── perturbations.py         # blur(), low_light(), occlude(), contrast_shift(), paste_patch()
    │   ├── metrics/
    │   │   ├── descriptive_accuracy.py
    │   │   ├── sparsity.py
    │   │   ├── stability.py
    │   │   ├── efficiency.py
    │   │   ├── robustness.py
    │   │   └── completeness.py
    │   └── viz.py                  # overlay_boxes(), overlay_heatmap(), save_figure()
    ├── tests/                      # pytest — pure-logic functions only, no GPU/network needed
    │   ├── test_data.py            # classify_image() on synthetic rows
    │   ├── test_regions.py         # mask_region/iou on synthetic arrays
    │   ├── test_metrics_sparsity.py
    │   ├── test_metrics_stability.py
    │   └── test_perturbations.py
    ├── scripts/                    # CLI entry points, one per pilot day, import from src/xai_pilot
    │   ├── 01_check_environment.py
    │   ├── 02_select_samples.py
    │   ├── 03_run_baseline_inference.py
    │   ├── 04_extract_regions.py
    │   ├── 05_descriptive_accuracy.py
    │   ├── 06_visual_sparsity.py
    │   ├── 07_stability_test.py
    │   ├── 08_efficiency_test.py
    │   ├── 09_robustness_test.py
    │   ├── 10_bounded_completeness.py
    │   └── 11_make_figures.py
    ├── notebooks/                  # thin, visual-only: load a results CSV, render images/plots
    │   ├── 02_sample_inspection.ipynb
    │   ├── 03_baseline_review.ipynb
    │   ├── 04_region_review.ipynb
    │   ├── 06_sparsity_review.ipynb
    │   ├── 07_stability_review.ipynb
    │   ├── 09_robustness_review.ipynb
    │   └── 11_figures_and_summary.ipynb
    ├── data/                       # gitignored: cached HF dataset, pilot_samples.csv
    ├── results/                   # tracked: small CSVs + final figures only
    └── report/
        ├── pilot_report_draft.md  # Day 12
        └── one_pager_summary.md   # Day 14
```

**Key interfaces used across days (so later tasks don't redefine things):**
- `data.classify_image(row) -> tuple[str, list[int]]` — `(primary_class, violated_rule_ids)`, `primary_class ∈ {"compliant","ppe_violation","fall_hazard","struck_by_risk"}`.
- `inference.AnswerResult` — dataclass: `answer: str`, `boxes: list[tuple[float,float,float,float]]`, `confidence: float`, `inference_ms: float`.
- `regions.mask_region(image: PIL.Image, box, mode: Literal["black","blur"]) -> PIL.Image`.
- `regions.iou(box_a, box_b) -> float`.
- `attribution.attention_rollout(model, inputs) -> np.ndarray` (HxW attribution map, values in [0,1]).

## Known Technical Risks & Mitigations (consolidated)

| Risk | Mitigation | Where handled |
|---|---|---|
| `torch` is CPU-only despite GPU present | Reinstall CUDA build matching driver (cu124/cu128 wheel) | Day 1 |
| `flash_attn` import crash on Windows | Monkey-patch `get_imports`, force `attn_implementation="sdpa"` | `model.py`, Day 1 |
| Florence-2 has no native VQA | Grounding-based proxy (Design Decision #1) | `inference.py`, Day 3 |
| Deterministic decoding makes naive stability test vacuous | Sample-based reruns + labeled deterministic reference | Day 7 |
| LRP/iNNvestigate don't run on transformers | Attention rollout + Captum gradient×attention instead | `attribution.py`, Day 3 |
| Repeated gradient extraction can exhaust memory (documented Abdallah failure mode) | Bound to top-1/top-2 regions only, batch size 1, release tensors between samples | Day 5, Day 10 |
| Visual vs. text attribution live in different mathematical spaces | Compute and report sparsity/stability separately per stream; report cross-modal agreement as a bridge metric, not a merged score | `metrics/sparsity.py`, Day 6, Day 11 |
| Grounding boxes may be missing for some images | Fallback to 4×4 grid regions for masking when no box returned | `regions.py`, Day 4 |

## Day-by-Day Plan

### Day 1 — Environment & Verified Access
**Goal:** environment is reproducible and every external dependency is confirmed working, not assumed.
- Copy this entire plan document into the repo as `Explainable-AI-Mustafa-Abdallah/pilot-plan.md` (committed to git, sitting alongside `workflow.md` and `two-week-pilot-plan.md`) so it's durably accessible in the project itself, not only in Claude Code's internal plans folder (`C:\Users\Scarecrow\.claude\plans\hidden-hopping-lecun.md`, which still has the original copy). This is the first thing committed.
- `git init` in `Explainable-AI-Mustafa-Abdallah/`; add `.gitignore` (`pilot/data/`, `pilot/.venv/`, `*.pyc`, notebook checkpoints).
- Create `pilot/.venv` (isolated from the global Python 3.10.11 install that other unrelated projects on this machine share).
- `pilot/requirements.txt`: `torch` (CUDA build), `transformers==5.2.0`, `datasets==4.5.0`, `accelerate`, `captum`, `huggingface_hub`, `opencv-python`, `Pillow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-image`, `scipy`, `pytest`, `jupyter`.
- Reinstall CUDA-enabled torch: `pip install torch --index-url https://download.pytorch.org/whl/cu124` (verify against current driver's supported CUDA at install time), confirm with `torch.cuda.is_available() == True`.
- `scripts/01_check_environment.py`: loads `microsoft/Florence-2-base-ft` with the `flash_attn`-patch from `model.py`, runs `<CAPTION>` on one sample image from the dataset, prints the caption and confirms it ran on GPU (`model.device.type == "cuda"`).
- Confirm dataset load: `datasets.load_dataset("LouisChen15/ConstructionSite", split="test")` — already known to work given the verified gated access above; this step just exercises the actual `datasets` library path (parquet streaming) rather than the HF API used for verification.
- **Deliverable:** `pilot/` git repo with first commit; `python scripts/01_check_environment.py` prints a real caption for a real downloaded image, on GPU.
- **Verification:** run the script, confirm exit code 0 and that the printed device is `cuda`.

### Day 2 — Sample Selection
**Goal:** a fixed, balanced, reproducible pilot subset.
- `data.py`: `classify_image(row)` reading `rule_1..4_violation`, returning `(primary_class, violated_rule_ids)` per the priority order in Design Decisions (rule 1 > rules 2/3 > rule 4 > compliant); `select_balanced_sample(ds, n_per_class=50, seed=42)` — 200 samples total (50 × 4 classes), drawn from the `test` split (3,004 images, ample headroom).
- `prompts.py`: `RULE_QUERIES` — 2 phrasings per rule (8 total, within the "5-10" range from the spec), e.g. Rule 1: `"hard hat"` / `"high-visibility vest"`; Rule 2: `"safety harness"` / `"fall-protection lanyard"`; Rule 3: `"guardrail"` / `"edge protection barrier"`; Rule 4: `("worker", "excavator")` proximity pair.
- `scripts/02_select_samples.py`: runs `select_balanced_sample`, writes `pilot/data/pilot_samples.csv` (`image_id, split, primary_class, violated_rule_ids`) and `pilot/data/safety_prompts.json`.
- `tests/test_data.py`: feed `classify_image` four synthetic rows (one per class) and one compliant row, assert correct `primary_class`.
- **Deliverable:** `pilot_samples.csv`, `safety_prompts.json`.
- **Verification:** `pytest tests/test_data.py -v` passes; `pd.read_csv("pilot_samples.csv").primary_class.value_counts()` shows ~50/class.

### Day 3 — Baseline Florence-2 Inference
**Goal:** real model outputs for every pilot sample, including the grounding-proxy VQA.
- `model.py`: `load_florence2()` (flash_attn patch + sdpa), `run_task(image, task_token, text_input=None) -> (text, parsed)` wrapping `processor(...)` → `model.generate(..., output_scores=True, return_dict_in_generate=True, max_new_tokens=1024)` → `processor.post_process_generation(...)`.
- `inference.py`: `answer_rule(image, rule_id) -> AnswerResult` — implements Design Decision #1 (open-vocabulary detection for rules 1-3, worker/excavator proximity check for rule 4); also calls `<MORE_DETAILED_CAPTION>` once per image for context, logged but not scored.
- `scripts/03_run_baseline_inference.py`: iterate `pilot_samples.csv` × assigned rule prompt, call `answer_rule`, time each call, write `pilot/results/baseline_predictions.csv` (`image_id, rule_id, answer, boxes, confidence, inference_ms`) and save 15-20 sample images with boxes drawn (via `viz.overlay_boxes`) to `pilot/results/figures/baseline/` for manual sanity-checking.
- **Deliverable:** `baseline_predictions.csv`, `results/figures/baseline/*.png`.
- **Verification:** open 10-20 saved overlay images in `notebooks/03_baseline_review.ipynb` and manually confirm boxes look plausible (a hard-hat box roughly on a head, not on a crane). Document a few failure cases — they matter for the Day 11 analysis.

### Day 4 — Grounding & Region Extraction
**Goal:** every sample has a standardized set of "explanation candidate regions," regardless of whether Florence-2 returned a box.
- `regions.py`: `mask_region(image, box, mode)` (black-fill or Gaussian blur, ksize=51), `grid_fallback_regions(image, grid=(4,4))`, `iou(box_a, box_b)`, `standardize_regions(answer_result, image_size) -> list[Region]` — uses the model's own returned box(es) when present, else the 4×4 grid.
- `scripts/04_extract_regions.py`: run `standardize_regions` over all Day-3 outputs, save before/after masked-image pairs for ~10 samples to `results/figures/regions/`.
- `tests/test_regions.py`: `iou` on known overlapping/non-overlapping synthetic boxes; `mask_region` on a synthetic uniform-color image, assert the masked region's mean pixel value changed and the rest didn't.
- **Deliverable:** region-extraction pipeline (importable functions), example masked images.
- **Verification:** `pytest tests/test_regions.py -v`; visually confirm a couple of before/after pairs in `notebooks/04_region_review.ipynb`.

### Day 5 — Descriptive Accuracy
**Goal:** does masking the top region actually change Florence-2's answer?
- `metrics/descriptive_accuracy.py`: `evaluate(image, rule_id, baseline: AnswerResult, top_regions) -> dict` — masks top-1 (then top-2) region, reruns `answer_rule`, returns `answer_changed: bool`, `confidence_drop: float`.
- `scripts/05_descriptive_accuracy.py`: run over all 200 samples, write `results/descriptive_accuracy.csv`; report `descriptive_accuracy_rate = mean(answer_changed)` overall and per class.
- **Deliverable:** `descriptive_accuracy.csv`, a results table/plot in `results/figures/`.
- **Verification:** spot-check 5 samples by eye — does the masked region plausibly contain the safety-relevant object, and did the answer change make sense?

### Day 6 — Visual Sparsity
**Goal:** is the explanation focused or scattered?
- `attribution.py`: `attention_rollout(model, inputs)` first (cheaper, no gradients); `gradient_x_attention(model, inputs)` via Captum only if attention rollout proves numerically stable on Florence-2's vision encoder (per the bounded-risk approach in `technical-limitations-take-and-mitigation.md` — do not promise both, ship whichever is stable, document which one ended up used).
- `metrics/sparsity.py`: `topk_mass_ratio(attr_map, k)`, `regions_above_threshold(attr_map, thresh)` — computed on the visual stream only (text-prompt-token sparsity, if attempted, is a separate, independently-reported number per Design Decision in the risk table — not merged).
- `scripts/06_visual_sparsity.py`: compute over all samples, write `results/visual_sparsity.csv`, save 5-10 example heatmap overlays (`viz.overlay_heatmap`).
- **Deliverable:** `visual_sparsity.csv`, example heatmaps comparing a focused vs. scattered case.
- **Verification:** `notebooks/06_sparsity_review.ipynb` — visually confirm the heatmap actually concentrates on something sensible for at least the better-scoring examples.

### Day 7 — Stability Test
**Goal:** genuinely repeatable explanations, not a trivially-perfect deterministic rerun (see Design Decision #2).
- `metrics/stability.py`: `run_n_times(image, rule_id, n=3, sample=True, temperature=0.7) -> list[AnswerResult]`; `region_overlap_score(results) -> float` (mean pairwise IoU of top-1 box across runs, or top-k grid-cell overlap when using the grid fallback); also run once with deterministic beam search and label it separately as the decoding-ceiling reference.
- `scripts/07_stability_test.py`: run over all samples, write `results/stability.csv` with both the sampled-stability score and the deterministic-ceiling flag.
- **Deliverable:** stability score table, a couple of stable vs. unstable example overlays.
- **Verification:** confirm the sampled runs actually differ at least sometimes (if every sampled run is identical too, temperature/sampling config needs adjusting — don't ship a vacuous result).

### Day 8 — Efficiency Test
**Goal:** practical runtime budget, with the 1,000-sample extrapolation the spec asks for.
- `metrics/efficiency.py`: `aggregate_timings(df) -> dict` (mean/median/max of `inference_ms`, masking time, re-inference time, attribution time, from columns already logged in Days 3/5/6/7's output CSVs — no new instrumentation needed, this day is aggregation, not re-running inference).
- `scripts/08_efficiency_test.py`: read the existing CSVs, compute aggregates, extrapolate `n_samples=1000` by linear scaling of the mean per-sample time, write `results/efficiency_summary.csv` + a runtime bar/box plot.
- **Deliverable:** `efficiency_summary.csv`, runtime plot.
- **Verification:** sanity-check the 1000-sample extrapolation against wall-clock time actually observed for the 200-sample runs in Days 3-7 (order-of-magnitude check, not a strict test).

### Day 9 — Robustness Test (+ bounded stretch task from `workflow.md`)
**Goal:** do explanations survive noise, and (time-permitting) can a fake PPE patch fool the model?
- `perturbations.py`: `blur(image, ksize)`, `low_light(image, gamma)`, `occlude(image, frac=0.2)`, `contrast_shift(image, factor)` — Level 1. A small Level 2: reworded/padded prompt variants already enumerated in `prompts.py`'s second phrasing per rule.
- `metrics/robustness.py`: `evaluate(image, rule_id, perturbation_fn) -> dict` — reruns `answer_rule` and attribution on the perturbed image, compares `answer_changed`, grounding-box IoU vs. baseline, and top-region drift.
- `scripts/09_robustness_test.py`: run Level 1 + Level 2 over all 200 samples → `results/robustness.csv`.
- **Stretch (bounded, time-boxed, can be dropped without affecting the day's core deliverable):** `perturbations.paste_patch(image, box, patch_image)` pasting a synthetic hard-hat sticker onto 20 `ppe_violation` images; rerun; check whether Florence-2 flips to "compliant" and whether the attribution method correctly highlights the pasted patch as the reason. This is `workflow.md` Step 5, scoped down to fit Day 9's time budget rather than promised as a guaranteed deliverable.
- **Deliverable:** `robustness.csv`, before/after examples; if time allows, the 20-image patch-test results as a labeled "stretch" subsection.
- **Verification:** visually confirm at least one clear success and one clear failure case for the report's "what worked, what failed" narrative (Day 14 needs this either way).

### Day 10 — Bounded Completeness
**Goal:** are the top-ranked regions actually necessary, focused on hard cases.
- `metrics/completeness.py`: `classify_sample(image, rule_id, baseline, top_regions) -> Literal["explanation_supported","explanation_weak","no_usable_explanation"]` per the bounded definition in `technical-limitations-take-and-mitigation.md` (mask top-1, optionally top-2, check answer change; no grounding box at all → `no_usable_explanation`).
- `scripts/10_bounded_completeness.py`: run over all samples, with extra attention to hard subsets already taggable from existing columns: `quality_of_info == "poor info"`, multi-rule-violation images flagged in Day 2's `violated_rule_ids`, and samples whose Day-3 answer required the grid fallback (no native box). Write `results/bounded_completeness.csv`.
- **Deliverable:** completeness results table, explicit failure-case examples for the hard subsets.
- **Verification:** confirm the hard-subset failure rate is reported separately from the overall rate (this is the number Professor Abdallah will care about most).

### Day 11 — Analysis & Figures
**Goal:** turn the six CSVs into professor-ready evidence.
- `notebooks/11_figures_and_summary.ipynb` + `scripts/11_make_figures.py`: baseline examples, masked-region examples, robustness examples, a single metric-summary bar chart across all six metrics, broken down by class.
- `results/pilot_metric_summary.csv`: one row per metric with overall + per-class numbers, pulled from Days 5-10's CSVs (no new computation, this is aggregation/presentation).
- Identify and write up the 3-5 most important failure patterns observed across Days 3-10 (e.g., grid-fallback rate, hardest class, weakest metric).
- **Deliverable:** `results/figures/` final set, `pilot_metric_summary.csv`.
- **Verification:** every number in the summary CSV should trace back to a specific Day 5-10 CSV — no recomputation, just rollup, so there's nothing new to verify beyond consistency.

### Day 12 — Draft Pilot Report
**Goal:** the technical memo.
- `report/pilot_report_draft.md`: 4-6 pages — motivation, dataset (with the real schema/rule mapping from "Verified Ground Truth" above), model (Florence-2 + the grounding-proxy adaptation, stated explicitly as an adaptation, not hidden), methods, the six metrics with actual numbers from Day 11, limitations (explicitly: 200/3,004 samples, Level 1+ partial Level 2 robustness only, bounded completeness, no LRP/DeepLIFT), next steps.
- **Deliverable:** report draft.
- **Verification:** every claim in the report should cite a specific CSV/figure produced in Days 1-11 — no invented numbers.

### Day 13 — Polish and Validate
**Goal:** make it presentable and reproducible.
- Clean tables/figures; **re-run the full pipeline end-to-end on a fresh 20-sample subset from scratch** (`pip install -r requirements.txt` in a clean venv → `01` through `11` scripts in order) to confirm it actually reproduces, not just "ran once on my machine."
- `report/one_pager_summary.md`: one-page professor summary + talking points.
- **Deliverable:** final report, one-page brief.
- **Verification:** the from-scratch 20-sample rerun completing without manual intervention is the actual test here.

### Day 14 — Meeting Package
**Goal:** assemble everything for the Professor Abdallah meeting.
- Finalize: one-page pitch, pilot report, sample visualizations, a short task list for the full study (full LRP/DeepLIFT exploration, full robustness Level 3, full-dataset completeness — explicitly framed as "what we'd want your guidance on").
- Prepare 5-minute talking points: what was tested, what worked, what failed, where Abdallah's expertise is needed next (LRP/DeepLIFT feasibility on transformers, statistical rigor for the metrics).
- **Deliverable:** complete collaboration package in `pilot/report/`.
- **Verification:** none beyond a final read-through — this is a writing/assembly day, not a code day.

## Out of Scope for This Pilot (explicit, matches `technical-limitations-take-and-mitigation.md`)
- Full LRP/DeepLIFT on Florence-2.
- Full adversarial model retraining (biased/adversarial weight training) — only input-level perturbation/patching.
- Full CMA temporal/video integration or SODA benchmark.
- Completeness or robustness over the full 3,004-image test split.
- The outreach email and 1-pager pitch from `Step-by-step-guide.md` (separate, later conversation per your answer above).

## Overall Verification Strategy
- `pytest pilot/tests/ -v` covers all pure-logic functions (`classify_image`, `iou`, `mask_region`, sparsity/stability formulas, perturbation functions) with synthetic inputs — these need no GPU, no network, run in seconds, and should pass before any day's script is trusted.
- Every day that touches the model produces a CSV (machine-checkable) and a handful of saved images (human-checkable in a notebook) — both are required before moving to the next day, since a silently-wrong region or a silently-broken grounding call would invalidate every later metric.
- Day 13's from-scratch rerun is the only end-to-end "integration test" — appropriate for a research pilot rather than mocking the model in CI.
