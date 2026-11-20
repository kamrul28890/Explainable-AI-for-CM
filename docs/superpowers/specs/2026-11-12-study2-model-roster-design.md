# Study 2: model roster for the comparison arm

**Date:** 2026-11-12. **Status:** proposed, awaiting the owner's approval of the roster.

**Goal:** run the Study 2 battery on 12–15 vision-language models, so the findings can be read as
claims about VLMs rather than about Qwen2.5-VL-7B.

## Constraints

Each candidate must satisfy four things.

1. **Open weights.** Attention (Stage 5), token ablation (Stage 4 "tokens") and completeness (Stage 9)
   need the model's internals.
2. **Fits one A40 (48 GB)** on RCAC Scholar, quantised where needed.
3. **Loads in Hugging Face transformers.** One backend protocol (`backends/base.py`) and one metric
   harness, so a difference between models is never a difference in the measuring instrument.
4. **Precedent.** Used in the academic literature, ideally on construction safety or on this dataset.

On this dataset:

- The dataset paper (Chen & Zou, arXiv 2508.11011) evaluated LLaVA-1.5 7B/13B, LLaVA-1.6 34B,
  MiniGPT-4 v2 and GPT-4V/4o.
- Sahraoui (arXiv 2511.15720) evaluated Molmo-7B and Qwen2-VL-2B.

Elsewhere in construction safety:

- Adil et al. (2025) evaluated GPT-4o, Gemini, Llama 3.2 and InternVL2.
- Adil et al. (2026) evaluated Gemma-3 4B, Qwen3-VL 2B/4B and InternVL3 1B/2B.

## The roster (15)

| # | Tier name | Model | Size | Why | Box output | Env |
|---|---|---|---|---|---|---|
| 1 | `qwen7b-4bit` | Qwen2.5-VL-7B, nf4 | 7B | primary arm (done) | pixels, model-input frame | A |
| 2 | `qwen7b-bf16` | Qwen2.5-VL-7B, bf16 | 7B | quantisation arm (done) | same | A |
| 3 | `qwen3b-bf16` | Qwen2.5-VL-3B | 3B | size, down | same | A |
| 4 | `qwen32b-4bit` | Qwen2.5-VL-32B, nf4 (~20 GB) | 32B | size, up | same | A |
| 5 | `qwen2-2b` | Qwen2-VL-2B | 2B | prior generation; used on this dataset | 0–1000 | A |
| 6 | `qwen3vl-8b` | Qwen3-VL-8B-Instruct | 8B | next generation, same lab | 0–1000 | B |
| 7 | `qwen35-9b` | Qwen3.5-9B (thinking off) | 9B | newest Qwen (Feb 2026) | to verify at Stage 0 | B |
| 8 | `internvl35-8b` | InternVL3.5-8B-HF | 8B | other lab, InternViT encoder; used in construction papers | 0–1000 | B |
| 9 | `glm41v-9b` | GLM-4.1V-9B-Thinking | 9B | other lab; grounding with boxes | 0–1000 | B |
| 10 | `gemma4-12b` | Gemma 4 12B-it | 12B | Google; no separate vision encoder | to verify at Stage 0 | B |
| 11 | `minicpmv45-8b` | MiniCPM-V 4.5 | 8B | widely used open VLM | to verify at Stage 0 | B (custom code) |
| 12 | `llava15-13b` | LLaVA-1.5-13B | 13B | the dataset paper's open baseline | 0–1 normalised | A |
| 13 | `llavaov-7b` | LLaVA-OneVision-7B | 8B | standard academic baseline | 0–1 normalised | A |
| 14 | `molmo-7b` | Molmo-7B-D | 8B | used on this dataset | points, not boxes | A (custom code) |
| 15 | `llama32v-11b` | Llama-3.2-11B-Vision | 11B | used in construction papers | none: verdict only | A (gated licence) |

### Reserves

- Pixtral-12B and Phi-4-multimodal, if a roster model fails Stage 0.
- Closed APIs (GPT-4o / Gemini), the dataset paper's comparison. These would give verdict and box
  only, cost money and send images to a third party, and run only if the owner asks.

## Design decisions

### D1. Every backend returns regions in original-image pixels

Found 2026-11-12. Qwen2.5-VL gives boxes in the frame of the image it was fed after the processor
resizes it (1200×900 becomes 812×588), not the original image. The harness read them as
original-image pixels.

Measured over the 1,187 correct-YES rows with a box (4-bit):

| | mean IoU | IoU ≥ 0.5 |
|---|---|---|
| frame as the harness read it | 0.058 | 2.1% |
| rescaled to the original frame | 0.389 | 40.6% |

bf16 is the same: 0.060 becomes 0.398.

So each backend converts its model's native frame to original pixels:

- pixels in the model-input frame;
- 0–1000;
- 0–1;
- points.

Each backend has a round-trip unit test on a known box. The fix to the existing Qwen arm is its own
item (see "Order"), behind a flag, with the frozen files kept.

### D2. Same question, native coordinate convention

All models get the same rule question and the same REASON / REGION / ANSWER format. The only
difference is the unit REGION is asked in, which matches what each model was trained on. Asking a
0–1000 model for pixels would measure format compliance, not localisation. The convention is part of
each tier's spec and is recorded in every results file.

### D3. Same pixels in

Every model receives the same image, downscaled to at most 501,760 px (the 640-block budget). Its own
processor does the rest. Each model's vision-token count is recorded per row.

Fixed-resolution models (LLaVA-1.5 at 336 px) see less detail. That is a property of the model and is
reported, not corrected.

### D4. Capability tiers decide which stages run

| Capability | Models | Stages |
|---|---|---|
| Box + attention + token ablation | Qwen family, InternVL, GLM-4.1V, LLaVA-OV, LLaVA-1.5, Gemma 4 (if Stage 0 passes) | all |
| Box only (attention-to-pixel map not yet built) | any of the above until its map is tested | 2, 3, 4 (black fill), 6, 7, 8, 10 |
| Point only | Molmo | 2, 6, 7; "point inside the annotated box" reported separately |
| Verdict only | Llama 3.2 Vision | 2, 6, 7 |

A stage a model cannot run is reported as not applicable, never as zero.

### D5. Two environments, and a parity check

- **Env A** is the frozen `transformers==4.49.0`, so the frozen Qwen2.5 results never move.
- **Env B** pins a current transformers release, needed by Qwen3-VL, Qwen3.5, InternVL3.5-HF,
  GLM-4.1V and Gemma 4.

Before any env-B model runs, `qwen7b-bf16` is re-run in env B on the existing parity manifest. The
paper then states how far the library change alone moves the replies.

### D6. Thinking models

Qwen3.5 and GLM-4.1V reason before answering. Where the model allows it, thinking is switched off, so
every model answers under the same 128-token budget.

If thinking cannot be switched off, the model gets a larger budget. Both the budget and the reasoning
length are recorded, and its latency (Stage 7) is reported apart.

### D7. Pins

Every model is pinned by commit SHA in `requirements/models.lock` before it downloads, as now.

## Order

Each step is one at a time, with GPU validation.

1. **Fix the coordinate frame (D1) for Qwen2.5-VL, behind a flag.**
   - Stage 3 and Stage 10 are re-scored on CPU from the frozen replies.
   - Stages 4, 5, 8 and 9 used the misplaced regions and are re-run on Scholar for both tiers.
   - Frozen files stay; new files are suffixed.
   - F3, F4 and F6 are re-read.
2. **Generalise the backend.** One Hugging Face VLM base class, with per-family adapters for:
   - prompt building;
   - coordinate frame;
   - image-token positions and patch grid.
3. **Build env B on Scholar and run the parity check (D5).**
4. **Stage 0 gate for each new model,** cheapest first. Each gets a GO / NO-GO and a capability tier.
5. **Stage 2 (accuracy) for every model that passes.** It is the cheapest and most comparable figure.
6. **Explanation stages** per capability tier.

## Cost

- **Time.** About one A40 day per model for the full battery at the 7B scale. The 32B model is about
  three times that. Thirteen new models is about two to three weeks of chained Scholar jobs.
- **Disk.** Weights total about 250 GB on scratch.
