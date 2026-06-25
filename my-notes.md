1. What is flash_attn?
It's a third-party Python package (flash-attn) implementing FlashAttention, a CUDA-kernel-level rewrite of the attention operation used inside every transformer layer. Standard attention computes a full seq_len × seq_len score matrix in GPU memory, applies softmax, then multiplies by values — this is memory-bandwidth-heavy. FlashAttention fuses the whole computation into tiles processed in fast on-chip SRAM, so it never materializes the full matrix in slower HBM memory. The math is the same; the kernel is just faster and uses less memory.

It ships as a package with precompiled CUDA extensions, built against specific (CUDA version, PyTorch version, Python ABI) combinations — and only as prebuilt wheels for Linux. There is no official Windows wheel; building it yourself on Windows requires a matching nvcc/MSVC/CUDA toolchain and routinely fails.

2. What does it do for our work, and why is it "necessary"?
It isn't actually necessary for our work — that's the key point. Florence-2's custom modeling file (pulled in via trust_remote_code=True) supports three interchangeable attention backends: eager, sdpa, and flash_attention_2. The file unconditionally has import flash_attn near the top, regardless of which backend you intend to use. So the import exists "just in case," not because our pipeline needs FlashAttention specifically. We never wanted the FlashAttention backend — we explicitly select sdpa instead (day01_environment.tex:34).

3. What is monkeypatching?
Monkeypatching is replacing a function/attribute at runtime, from outside the module that defines it, without editing that module's source. In Python this works because a function is just an object bound to a name in some module's namespace — you can rebind that name to point somewhere else while your process is running.

Generic example:


import math
math.sqrt = lambda x: -1   # monkeypatch: anyone calling math.sqrt now gets -1
Nothing in math's source changed; you redirected the name from outside.

4. What our specific monkeypatch did
Look at model.py:13-31:


from transformers.dynamic_module_utils import get_imports

def _get_imports_without_flash_attn(filename):
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

@contextmanager
def _no_flash_attn():
    with patch("transformers.dynamic_module_utils.get_imports", _get_imports_without_flash_attn):
        yield
Background on what get_imports does: when you load a model with trust_remote_code=True, before transformers even executes the model's custom Python file, it statically scans that file's import statements (via regex, not execution) and builds a list of "this file needs these packages." It then checks each is installed, and raises ImportError before the model loads at all if one is missing. Florence-2's file has flash_attn in that scanned list, so on a machine without flash_attn installed, loading crashes immediately — you never even reach the point of choosing an attention backend.

_get_imports_without_flash_attn calls the real get_imports, then deletes "flash_attn" from the returned list. _no_flash_attn() is a context manager that, for the duration of the with block, replaces transformers.dynamic_module_utils.get_imports with this filtered version using unittest.mock.patch. Then in model.py:39-45:


with _no_flash_attn():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=dtype,
    ).to(device)
Two distinct fixes are stacked here:

The monkeypatch fools the pre-flight import check into never flagging flash_attn as required, so from_pretrained doesn't abort with ImportError.
attn_implementation="sdpa" separately tells Florence-2's forward pass to actually use torch.nn.functional.scaled_dot_product_attention — PyTorch's own built-in fused attention kernel — instead of trying to instantiate or call into flash_attn at runtime.
Without (2), even if (1) let loading succeed, the model might still try to use the FlashAttention code path during inference and fail. Without (1), you never get past loading. Together: flash_attn is never imported, never required, never executed — the model runs entirely on PyTorch's native SDPA kernel instead.

5. Would this happen on Linux too?
Mostly no, but conditionally yes. On Linux, pip install flash-attn generally works (prebuilt wheels, or a build that isn't fighting Windows' toolchain mismatch), so you could just install it and pass attn_implementation="flash_attention_2" directly — no patch needed. The report explicitly calls this a Windows-specific workaround for that reason (day01_environment.tex:128-131).

That said, the underlying trigger isn't "Windows" per se, it's "flash_attn not installed" — which can also happen on Linux (e.g., older GPU architecture unsupported by FlashAttention, or a deliberate choice to skip it). In that case the identical ImportError and identical fix would apply on Linux too. Windows is just the platform where it's essentially guaranteed.

6. Does this affect our results or performance?
Outputs/correctness: no. SDPA and FlashAttention compute the same mathematical operation (scaled dot-product attention + softmax). They can differ in floating-point rounding at the last bits due to different kernel tiling/summation order, but that's the same order of nondeterminism already discussed in the reproducibility chapter (day13_reproducibility.tex) — not a source of different captions, regions, or conclusions.

Speed/memory: a real but probably small effect. FlashAttention's advantage shows up most for long sequences and large batches, where avoiding the full attention matrix saves a lot of memory bandwidth. Florence-2-base-ft here runs on single images with short prompts and short generated captions — a regime where SDPA (itself a fused, optimized kernel in modern PyTorch) is already close to FlashAttention's speed. Nothing in the rest of the report (including the efficiency chapter, which doesn't mention sdpa or flash_attn at all) flags attention-backend choice as a measured bottleneck — the Day 1 inference time reported was 1565.3 ms for one image, which the rest of the pilot just treats as a baseline, not something attributed to this choice.

So: the workaround is purely an engineering necessity to get the model to load on Windows at all — it doesn't change what the model computes or meaningfully change the pilot's downstream numbers.

2. Why Florence-2, and what else could we have used?
Straight from Research-prposal.md §5.2, the stated reasons were:

lightweight enough for a short pilot (231.6M params, MIT license, ungated — runs comfortably on the local RTX 3070's 8GB VRAM)
supports grounding/detection-style outputs natively, so a predicted region (a box) falls out of the model directly rather than needing a separate localization step
gives a practical, fast starting point for visual attribution work
The proposal explicitly names an alternative: Qwen2-VL, kept as "a second-stage comparison model... stronger for general VQA and broader visual reasoning," with Florence-2 picked as "the better first model for a fast feasibility test." (Side note: Qwen2.5-VL-3B and -7B-Instruct weights happen to be sitting in this machine's HF cache, but nothing in src/xai_pilot references Qwen — I'd treat that as leftover from other work on this machine, not evidence it was tried here.)

3. What would be different with another model?
This isn't hypothetical — Day 6 hit exactly this wall and documented it. Two concrete things would change:

a) No native VQA → had to fake it. Florence-2-base-ft has no <VQA> task token at all (pilot-plan.md) — so a safety question like "is the worker wearing a hard hat?" can't literally be asked. The workaround: convert each rule into an open-vocabulary detection query and infer compliant/violation from whether Florence-2 returns a box. A model like Qwen2-VL/LLaVA, which does support free-text VQA, would let you ask the question directly — but you'd lose Florence-2's clean "the explanation is literally a box" framing and need a separate method to recover which region a free-text answer was based on.

b) Attention rollout broke on Florence-2's architecture. day06_visual_sparsity.tex:31-53 found that Florence-2's encoder runs one shared self-attention stack over a fused image+text sequence (1 pooled token + 576 image patches + text prompt tokens, all mixed in the same attention matrix). Classic attention rollout assumes a single-modality stack, so it doesn't cleanly apply — they had to fall back to decoder→encoder cross-attention instead. A model with a cleanly separated vision-tower self-attention (image patches only) feeding into the language model via a separate cross-attention or projection step would not have hit this problem, and standard rollout could have been used as originally planned. Model size would also matter directly: Qwen2.5-VL-7B is roughly 30x larger than Florence-2-base-ft, which would change the efficiency numbers and likely force smaller batches or quantization.

4. Scope for the full paper — dataset and models
This is explicitly not decided yet — it's one of the open items the team is bringing to Professor Abdallah rather than something already settled. Two documents lay out the plan vs. the open questions:

Stated plan (Research-prposal.md §2.1):

ConstructionSite 10k — primary dataset, already used (10,013 images, 7,009 train / 3,004 test)
SODA — spatial-clutter/object benchmark, for robustness under varied layout/weather/viewpoint — not yet touched
CMA (Construction Meta Action) — video clips, for testing whether explanations hold up when the hazard depends on motion over time — not yet touched
Models: Florence-2 (done) + Qwen2-VL as the second-stage comparison model for broader VQA reasoning
Open decisions, explicitly flagged for Professor Abdallah (full_study_task_list.md):

whether decoder→encoder cross-attention is an acceptable stand-in for LRP/DeepLIFT, or a transformer-native method needs adding (item 1)
whether to scale to the full 3,004-image test split, or fix the struck_by_risk class shortage first (only 13 examples exist in the whole split) — item 4
SODA/CMA integration and a properly powered adversarial-robustness study are confirmed out of scope so far — "none of these were attempted or partially attempted."
So the honest answer is: the proposal's default scope is ConstructionSite (full) + SODA + CMA with Florence-2 + Qwen2-VL, but which of those actually gets built depends on decisions the pilot deliberately left for the upcoming meeting rather than pre-committing to.

1) The exact VQA workaround — with example
Florence-2's task vocabulary is closed-set tokens (<CAPTION>, <OD>, <OPEN_VOCABULARY_DETECTION>, etc.) — there's no token that means "answer this free-text question." So inference.py never asks a question at all. It converts each safety rule into one or more detection calls, and derives a yes/no answer from whether/where boxes land.

Concretely, for rule_1 ("does the worker have a hard hat?"), inference.py:81-119 does this:


def _answer_presence_rule(model, processor, image, rule_id, phrasing_index=0, **run_kwargs):
    phrase = RULE_QUERIES[rule_id][phrasing_index]          # "hard hat"
    worker_boxes, worker_conf = _detect(model, processor, image, "worker")
    object_boxes, object_conf = _detect(model, processor, image, phrase)

    distance_threshold = PRESENCE_PROXIMITY_FRACTION[rule_id] * max(image.width, image.height)
    covered = all_boxes_covered(worker_boxes, object_boxes, distance_threshold=distance_threshold)
    answer = "compliant" if covered else "violation"
    ...
Where _detect is just Florence-2's real <OPEN_VOCABULARY_DETECTION> task, called with a noun phrase as the prompt:


def _detect(model, processor, image, phrase, **run_kwargs):
    _, parsed, confidence = run_task(model, processor, image, "<OPEN_VOCABULARY_DETECTION>", text_input=phrase)
    return [tuple(b) for b in parsed["<OPEN_VOCABULARY_DETECTION>"]["bboxes"]], confidence
So "is the worker wearing a hard hat?" becomes, mechanically:

Detect "worker" → get N worker boxes.
Detect "hard hat" → get M hard-hat boxes.
For every worker box, check whether a hard-hat box is overlapping or within a small distance threshold (all_boxes_covered).
If every worker has a nearby hard hat → "compliant"; if any worker doesn't → "violation".
Note this isn't the first thing they tried — the original v1 (just "does a hard hat exist anywhere in the image?") was tried first and failed: it had only 3% sensitivity to real violations, because most multi-worker images had at least one compliant worker, so a hard-hat box almost always existed somewhere, regardless of the actual violation (day03_baseline_inference.tex:39-59). The per-worker coverage check (v2, shown above) was the fix. Worth knowing this if you're asked "did the workaround just work first try" — no, it failed quietly once before the redesign.

Rule 4 (struck-by risk) works the same way but checks proximity between two independently detected classes ("worker" vs "excavator") instead of presence-vs-coverage.

2) Do other models (Qwen2-VL etc.) have native VQA?
Yes — and this is the actual architectural distinction worth understanding. Florence-2 is a multi-task model trained on a fixed menu of vision tasks with discrete task tokens; it was never trained to follow open-ended instructions. Qwen2-VL/Qwen2.5-VL, LLaVA, InstructBLIP, GPT-4o, etc. are instruction-tuned chat models with a vision encoder bolted onto a decoder LLM — they're trained on exactly the kind of "here's an image, here's a free-text question, answer in free text" data. So with Qwen2.5-VL you could literally send:


image + "Is the worker in this image wearing a hard hat? Answer yes or no."
and get a direct answer, no grounding-proxy translation needed.

The catch for our purposes: Qwen2.5-VL's free-text answer doesn't automatically come with a region the way Florence-2's detection call does. To recover "what part of the image is this answer based on," you'd need a separate step — Qwen2.5-VL does support box/point grounding as its own capability (so you could ask it to also point at the hard hat), or you'd fall back to gradient/attention-based attribution on top of the VQA call. So it's a real tradeoff, not a strict upgrade: Florence-2 gives you the region for free but no native question-answering; Qwen2.5-VL gives you native question-answering but the region has to be earned separately.

3) Would attention rollout break the same way for other models?
Your intuition is right in spirit but the framing needs two corrections.

What actually generalizes: the failure isn't "any transformer breaks rollout" — it's specifically "any model where image and text tokens get mixed into one self-attention stack breaks rollout's interpretability." That's not unique to Florence-2. Modern VLMs that splice image patch tokens directly into the same token sequence as text — which includes LLaVA and the Qwen-VL family, just via a causal decoder rather than Florence-2's bidirectional encoder — have the same property: every layer's self-attention matrix mixes both modalities together, so rollout's recursive composition can't be cleanly read back as "pure image contribution." So yes, this would very likely recur with Qwen2.5-VL or LLaVA too, not just Florence-2.

What's wrong in the framing: the model didn't "fall back to a standard encoder-decoder to get output." The model's forward pass and its output (the caption, the detection box) were never affected by the rollout problem — Florence-2 produced its answers the same way the whole time. What changed was only the explanation method sitting on top of it: instead of rolling out the encoder's self-attention, Day 6 used the decoder's cross-attention into the encoder directly, which Florence-2 happens to expose because it's a genuine encoder-decoder architecture with a real cross-attention layer (day06_visual_sparsity.tex:31-59).

That last detail matters for your generalization question: that specific fallback (cross-attention) is only available because Florence-2 is encoder-decoder. LLaVA and the Qwen-VL family are decoder-only — there's no separate encoder/decoder cross-attention layer to read out. So if you swapped in Qwen2.5-VL and hit the same rollout problem, you would not have the same escape hatch; you'd need a different fix (raw self-attention from specific layers/heads, or a gradient-based method like Integrated Gradients on the vision tower, which is architecture-agnostic). So: the failure mode generalizes, but the fix does not — it's model-specific.

4) Recommended scope for the full paper
I checked the current literature before answering this, since "12+ impact factor, needs to be really good" changes the bar substantially from a pilot. Three things I found change my recommendation from "just follow the original proposal":

The space got crowded fast. Automation in Construction alone has published multiple VLM-safety papers in 2025 (Chan et al., AiC 177; Zhou et al., AiC 180; a safety-compliance VLM paper in AiC 179), and a January 2026 arXiv study already benchmarks GPT-4o, Florence-2, and LLaVA-1.5 head-to-head on construction-worker understanding. None of these, however, do a systematic faithfulness evaluation of the explanations themselves (I checked the closest-looking one, Clip2Safety — it's about detection speed/accuracy, not explanation quality). The six-metric XAI evaluation is still the open niche — that's your differentiator, and the scope should be built to maximize it, not just to scale up sample size.

Models — recommend three, not one:

Florence-2 (keep) — your instrumentable, region-native baseline.
Qwen2.5-VL (3B and 7B) — confirmed current state-of-the-art open-weight VLM with both native VQA and native grounding, so you can run the exact same six metrics against a model that doesn't need the grounding-proxy workaround, and report whether the proxy's biases (e.g. the worker-vs-PPE-box confound flagged in full_study_task_list.md item 3) are an artifact of Florence-2 specifically or recur anyway.
One closed frontier model (GPT-4o or equivalent) as a benchmark-only reference, not for deep attribution — reviewers in this space now expect a frontier-model comparison point (every recent paper I found includes one), even though you can't run gradient/attention methods on it through an API.
Datasets — keep the proposal's three, but justify each one's job precisely:

ConstructionSite 10k, scaled from 163 to the full 3,004-image test split — but pair this with the unresolved struck_by_risk imbalance (13 examples total) rather than silently inheriting it; a 12+ IF reviewer will catch an unpowered class immediately.
SODA — confirmed real and, notably, already published in Automation in Construction itself (19,846 images, 15 classes, multi-site/weather/viewpoint). Use it for out-of-distribution robustness/completeness — does the explanation degrade gracefully on a different site distribution, not just on perturbed versions of the same images.
CMA (Construction Meta Action) — confirmed real, also published in Automation in Construction (1,595 clips, 7 action classes). This is where I'd push hardest: extend the stability metric from "does the explanation survive stochastic reruns on one static image" (current pilot) to "does the explanation stay consistent frame-to-frame on the same hazard across a video clip." Nothing in the current literature does explanation-stability over time for construction VLMs — that's a genuinely new contribution, not just a bigger dataset.
One more thing I'd add to scope, prompted by your own internal task list: full_study_task_list.md item 1 already flags that cross-attention is standing in for LRP/DeepLIFT, undecided. For a top venue, I'd resolve that by adding one gradient-based method (Integrated Gradients via Captum, already a pinned dependency) as a second, independent attribution stream — so the paper can report whether cross-attention and IG agree, rather than resting faithfulness claims on a single method that was originally a fallback, not a deliberate choice.