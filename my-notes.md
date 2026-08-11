Chapter 1

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



Chapter 2

"struck_by_risk" — what it actually means
This is rule_4: a worker detected too close to an excavator — inside its blind spot or operating/swing radius — not a missing-equipment problem like the other three rules, but an unsafe spatial relationship between a person and moving heavy machinery (prompts.py:34: "worker proximity to excavator operating radius / blind spot").

Concretely, from the actual sample file, image 0000331 is one of these 13 (pilot_samples.csv): the dataset's own annotators marked it rule_4_violation, meaning a worker is positioned close enough to an excavator that a swing of the arm/bucket, or the operator's blind spot, could plausibly strike or pin them. Our pipeline tests this geometrically, not by asking a yes/no question: inference.py:122-144 detects "worker" boxes and "excavator" boxes independently, then flags "hazard" if any worker box overlaps or sits within a small distance threshold of an excavator box. There's no PPE item to find here — the hazard is the proximity itself.

"compliant" — what it actually means
This is the row where none of the four rule fields (rule_1_violation … rule_4_violation) are set in the dataset — every applicable safety requirement in that scene is satisfied: workers have the required PPE, anyone at height is tied off or protected by a guardrail, and no one is in an excavator's danger zone (data.py:34-35: if no violation field is set, classify_image returns "compliant").

Example from the CSV: image 0000057 is compliant, and was assigned to be tested against rule_1. That assignment is round-robin, not because rule_1 is "the" hazard in that photo — assign_rule_id cycles compliant samples evenly across all four rules specifically so each rule gets some "model correctly says compliant" test cases, rather than rule_1 hogging every compliant sample and rules 2-4 getting none (day02_sample_selection.tex:33-43). So "compliant + rule_1" means: this scene has no PPE violation, and Florence-2 is specifically being checked on whether it correctly avoids flagging one.

The four rules
Rule	What it checks	Hazard class it maps to	Query phrase(s) used
rule_1	Basic PPE present (hard hat, hi-vis vest)	ppe_violation	"hard hat" / "high-visibility vest"
rule_2	Fall protection (harness/lanyard) when working at height (≥3m)	fall_hazard	"safety harness" / "fall-protection lanyard"
rule_3	Edge protection (guardrails) at height/excavation edges (≥3m)	fall_hazard	"guardrail" / "edge protection barrier"
rule_4	Worker inside excavator's blind spot/swing radius	struck_by_risk	proximity pair ("worker", "excavator")
Notice rule_2 and rule_3 both map to the same class, fall_hazard — they're two different mitigations for the same hazard type (personal protection via harness vs. structural protection via guardrail), not two different hazards (config.py:19-24).

When a single image violates more than one rule, CLASS_PRIORITY = ["ppe_violation", "fall_hazard", "struck_by_risk", "compliant"] breaks the tie. Real example from the CSV: row 0001986 violates both rule_1_violation and rule_3_violation — but is classified ppe_violation, not fall_hazard, because rule_1's class sits earlier in the priority list. The image is still recorded as having violated rule_3 too (in violated_rule_ids), it just isn't the rule Florence-2 gets tested against for that sample.

How the rules drive the actual conclusion
This is a four-step chain, and it's worth being precise about which step does what:

Ground truth comes from the dataset, not from us. Each row already carries up to four annotator-written fields (rule_X_violation: {bounding_box, reason} or null). Day 2 just reads these into a class label — it doesn't run any model.
Each sample is assigned exactly one rule to be tested against (assign_rule_id) — the one unambiguous violated rule for violation classes, or a round-robin rule for compliant ones.
Florence-2 is run against that one rule (inference.py, Day 3) and produces a predicted answer (compliant/violation, or safe/hazard for rule_4) plus the grounding box(es) it used to get there.
Two separate comparisons happen on top of that: (a) predicted answer vs. the dataset's ground-truth class — is Florence-2 even right? — and (b) the six XAI metrics (descriptive accuracy, sparsity, stability, efficiency, robustness, completeness) applied to the grounding box itself — is the box Florence-2 used to get that answer actually trustworthy as an explanation (does masking it flip the answer, does it stay put on reruns, does it survive blur/occlusion, etc.)?
So the rules aren't just a labeling scheme — they're what turns each image into a specific, falsifiable test ("for this image, does Florence-2 correctly judge rule_3 compliance, and is the region it points to actually why it judged that way").

The core problem in Chapter 4
Chapter 4 has to answer "which region does the model's judgment depend on?" so Day 5 can mask it. But Florence-2's grounding output carries no saliency, attention, or importance signal — it just returns detection boxes. There is no principled "this box mattered most" score to rank by.

The resolution was a deliberate, documented compromise: rank candidate boxes by area, largest first (regions.py:135). "A bigger detection is more likely salient than a sliver." This was flagged as a stand-in, not a real importance ranking — and it's the single most consequential decision in the whole pilot.

Problems this caused (measured downstream, not just suspected)
1. Area-ranking systematically promotes the wrong entity — the central flaw.
For a PPE rule, the worker's whole-body box is almost always bigger than the hard-hat/harness box it's being compared against. So area-ranking masks the worker, not the PPE item. Chapter 10 measured it:

Worker ranked top-1 in 48/50 (96%) of rule_1 samples.
Masking the worker flips the answer only 37.5% of the time vs 49.4% for the actual safety object — an 11.9 pp gap that holds across the whole pilot.
rule_1 (the PPE rule the project cares about most) has the weakest completeness of any rule: 33.3% supported vs rule_3's 55.1%.
Because standardize_regions feeds Chapters 5, 6, 7, and 10 unchanged, this one function's bias contaminates four metrics at once.

2. The grid fallback is crude.
When Florence-2 returns zero boxes (4/163), the fallback masks the top-left cell of an unranked 4×4 grid — an essentially arbitrary corner, not a meaningful region. Fine as a rare placeholder (2.5%), but 3 of those 4 cases are rule_3 ("guardrail" is a hard open-vocabulary target), so it's not random — it concentrates exactly where grounding is weakest.

3. Near-duplicate boxes entangle the masking signal (worker-fallback artifact).
Florence-2 sometimes returns almost pixel-identical boxes for "worker" and "harness" (Ch10 traces image 0000642: worker (436,232,500,420) vs harness (439,232,495,420)). Masking the top-1 region then erases both, the worker vanishes, a degenerate fallback branch fires, and the answer changes — registering as a genuine "explanation supported" result when it isn't. This inflated the headline completeness number (71/163 → corrected 68/163).

4. Validation was thin. The masking primitive was visually sanity-checked on only 2 of 10 saved pairs — fine for a pilot, not for scale.

What scale-up needs (the one high-leverage fix)
This is decision 3 in the meeting package (day14, item 3) and Chapter 10's top recommendation: replace area-only ranking with rule-aware ranking — prefer a box whose label matches the rule's queried object class (hard hat, vest, harness, lanyard, guardrail, excavator) before falling back to area.

Why it's the right first move at scale:

No new model calls — it only re-ranks boxes Florence-2 already returned. Cheap and safe.
It comes with a falsifiable prediction: the 11.9 pp worker-vs-object gap should narrow substantially if the fix works.
It plausibly improves three metrics' weakest PPE results simultaneously.
Note the labels already exist in the pipeline — standardize_regions receives a labels list (04_extract_regions.py:57), it just ignores it for ranking. The fix is small.

Hindrances to scaling, and how to solve each
Hindrance	Why it bites harder at scale	Fix
Area-ranking bias (worker over PPE)	Every masking metric inherits it; on a full 3,004-image run it skews the headline numbers for the class you care about most	Rule-aware ranking (above). Highest priority.
No true importance signal even after the fix	Rule-aware ranking still isn't a real attribution; a paper reviewer may not accept area as "importance"	Rank regions by cross-attention mass from the Chapter 6 attribution instead of area — ties to decision 1 (attribution method). More work, but makes the ranking defensible.
Grid fallback is arbitrary + concentrated in rule_3	More no-box cases at scale, mostly rule_3	(a) Improve the "guardrail"/"edge protection" grounding prompt; (b) replace the top-left-cell hack with a proper occlusion-sensitivity sweep (mask each of the 16 cells and take the one that moves the answer most) so even no-box samples get a meaningful region.
Near-duplicate worker/object boxes → fallback artifact	Recurs whenever PPE is worn tight to the body; silently inflates completeness/robustness	Detect box near-duplication (high IoU between worker and object box) and handle explicitly — dedupe, or don't let masking one erase both; rule-aware ranking also partly defuses it by masking the object rather than the worker.
Black-patch masking is out-of-distribution	A solid black rectangle is itself a salient artifact the VLM never saw in training; at scale it can drive spurious answer flips	The primitive already supports mode="blur" (regions.py:57); run a mask-mode sensitivity check (black vs blur vs inpaint) before trusting flip rates on the full set.
Masking validated by eye on 10 samples	Can't eyeball 3,004	Add automated invariants/property tests (mask touches only the intended box, coordinates stay in-bounds, area monotonic) so correctness is checked, not inspected.
Streaming I/O from the Hub	GPU starved waiting on network per image	Already fixed — load_construction_site now reads the local parquet copy, so a cluster run is compute-bound, not I/O-bound.
One reassurance: efficiency is not a scaling obstacle. Chapter 8 measured the full six-metric pipeline at ~0.5 GPU-hours per 1,000 samples on a single RTX 3070. Every problem above is about validity, not compute — the ranking fix is the one that unlocks the most correctness for the least effort.

Chapter 5
The core idea, and the core problem
Descriptive Accuracy is a deletion test: mask the top-ranked region, re-run the proxy, and if the answer flips the region was load-bearing. The implementation (descriptive_accuracy.py) masks top-1, then top-1+top-2 cumulatively, and records whether each flips the baseline answer. Headline: 36.2% (top-1), 35.0% (top-1+2).

The core problem is that this metric isn't an independent measurement here — it's largely a second view of Day 3's proxy quality and Day 4's ranking bias. The per-rule flip ordering (rule_4 44.0% > rule_3 42.9% > rule_2 38.5% > rule_1 27.0%) tracks Day 3's sensitivity ordering almost exactly. A proxy that barely discriminates (rule_1, 16% sensitivity) has little for masking to disrupt, so it flips least — which looks like "poor descriptive accuracy" but is really "weak underlying proxy." The metric is confounded by the thing it sits on top of.

Problems this chapter surfaced (measured, not suspected)
1. The non-monotonicity anomaly — the headline finding. Masking more (top-1+2 = 35.0%) flips less than masking top-1 alone (36.2%). In 15/163 samples, adding the second mask un-flips an answer the first mask had flipped, restoring baseline. This should be impossible for a clean deletion test.

2. The mechanism: metric–model entanglement via the fallback branch. Traced concretely on image 0000069 (rule_1):

Worker box (10,778 px²) is 12× the hard-hat box (891 px²), so area-ranking makes the worker top-1.
Masking the worker → Florence-2 re-detects no worker → the if not worker_boxes fallback in _answer_presence_rule fires ("compliant if object exists, else violation"). Hat still visible → compliant (a flip from baseline violation).
Also masking the hat → still no worker, and now no object either → fallback answers violation = baseline again.
So the masking test manufactures the exact zero-worker condition the fallback was built for, and the answer then depends only on whether the other box is still visible — a binary, non-monotonic dependency. All 15 regressions follow this exact full-revert pattern.

3. Flips get mis-attributed. A flip can mean "removed load-bearing evidence" or "lost the worker detection and rerouted to a cruder branch." The raw 36.2% mixes both. Chapter 10 later quantified how much this contaminates related numbers (it corrected completeness 43.6%→41.7% for exactly this reason).

4. Inherited Chapter 4 bias. Because top-1 is usually the worker for PPE rules, rule_1's 27% is depressed by masking the wrong entity — the same area-ranking confound, propagated forward.

5. Coarse signal. The answer space is binary (compliant/violation), and Chapter 10 checked the confidence proxy as an alternative signal and found it non-discriminative (median drop 0.032 whether or not the answer changed). So there's no graded fallback — masking effects are all-or-nothing.

What scale-up needs
The single most important change is to instrument the metric to separate genuine flips from fallback-rerouting flips. Chapter 10 already proved the approach: an extra rerun set (159 reruns) that captures post-mask worker_boxes, so you can tell whether a flip co-occurred with the worker detection dropping out. At scale this should be baked into evaluate() itself — emit a flip_due_to_worker_loss flag and report a corrected descriptive-accuracy rate, not just the raw one.

Second, this metric is a direct beneficiary of the Chapter 4 rule-aware ranking fix — masking the actual PPE object instead of the worker both raises rule_1's flip rate and removes most of the worker-loss artifact at its source.

Hindrances to scaling, and how to solve each
Hindrance	Why it bites at scale	Fix
Fallback-rerouting confound	Raw flip rate over-counts/mis-attributes flips; untrustworthy on a full 3,004-image run	Instrument evaluate() to log post-mask worker_boxes and flag worker-loss flips; report corrected rate (Ch10's method, promoted into the metric)
Metric confounded by proxy quality	Low rule_1 flip rate reads as "bad explanation" when it's "weak proxy"	Fix ranking (Ch4); and/or restrict the metric to samples where the proxy is actually discriminating (e.g. baseline answer correct) so it measures fidelity, not proxy weakness
Non-monotonicity from branching logic	Recurs for any deletion test against a branching proxy (also hits Ch9, Ch10)	Add a monotonic-masking invariant that flags violations automatically; longer-term, make the proxy answer graded (fraction of workers covered) so effects are monotonic
Binary answer + dead confidence signal	No statistical power / no partial-effect detection at scale	Expose a continuous score from answer_rule (coverage fraction, min distance ratio) and measure descriptive accuracy as magnitude of change, not a boolean flip
Black-mask is out-of-distribution	A flip may be the model reacting to a black rectangle, not to missing evidence	Run mask-mode sensitivity (black vs blur — already supported — vs inpaint) on a subset; confirm flip rate is robust to mode
Small-n class rates	struck_by_risk's 61.5% is n=13, directional only (ties to the class-scarcity issue in the feedback doc you have open)	Decision 2 in the meeting package: set a minimum n per class before reporting a rate as stable
Figure selection bias	Only the first 10 flip cases are saved — no non-flip or regression cases	Save a balanced set including non-flips and the 15 regression cases for audit
Compute is not the blocker. This metric adds 2 reruns/sample and the corrected instrumentation adds ~1 more; Chapter 8's ~0.5 GPU-hours per 1,000 samples for the whole pipeline means even the instrumented version scales trivially. Every issue here is validity, not cost.

The through-line: Chapters 4, 5, 9, and 10 all trip over the same two mechanisms — area-ranking picking the worker, and masking-that-worker rerouting through the fallback branch. Fixing the ranking (Ch4 decision) plus instrumenting for worker-loss flips (Ch5/Ch10 method) addresses the root of all four chapters at once.

Chapter 6
What happened: the planned method didn't apply
The plan called for attention rollout (the textbook ViT attribution method). Before writing any code, the team checked Florence-2's actual modeling code and found rollout's precondition fails:

Florence-2's encoder is one shared self-attention stack over a fused image+text sequence — 577 tokens = [1 global pool token] + [576 image patches, 24×24] + [text tokens].
Rollout assumes a single-modality stack; composing Florence-2's layers would recursively mix in text-token attention with no way to separate the image-only part back out.
Substitution: decoder→encoder cross-attention (attribution.py) — a per-step, already-computed quantity that needs no layer composition and no single-modality assumption. This is genuinely good practice (verify before you build), but the consequence is that the attribution signal is a choice-among-constraints, not the textbook method — which matters for how the paper claims the framework ported.

Problems this chapter surfaced
1. The central finding: sparsity mostly measures target physical size, not explanation quality. The correlation between log(box area) and top-5 mass ratio is −0.70. Hard hat (small, 94k px²) scores 0.195; excavator (large, 288k px²) scores 0.079 — a 2.5× gap that looks like "hard-hat explanations are sharper" but is mechanical: a small object covers few of the 576 cells, so its mass is concentrated by construction. Any cross-rule sparsity comparison is invalid without controlling for object size.

2. Decoding-mode mismatch. Cross-attention is extracted with greedy decoding (num_beams=1), but the answers in Chapters 3–5 use beam search (num_beams=3). HF beam search reorders the batch dimension and the returned attentions aren't un-reordered, so beam attentions don't line up. Greedy sidesteps it — but that means the heatmap explains a fresh greedy rerun, not the actual beam-search answer, and this equivalence was spot-checked on only one image (0000007).

3. The off-by-one trap (caught). 577 isn't a perfect square; √577 ≈ 24.02. Naively reshaping would silently shift every heatmap by one cell with no error. The fix depends on the verified architecture fact — drop the global token (index 0) before reshaping to 24×24. Caught, but it's the kind of silent failure that recurs if the input resolution or patch grid changes at scale.

4. Heavy averaging diffuses the signal. The heatmap averages over decoder layers, heads, and all generated tokens (attribution.py:95-96). The top-5-of-576 mass ratio is only 0.108 — most mass is spread thin. Some of that "diffuseness" may be over-averaging washing out the informative heads/steps, not the model genuinely being unfocused.

5. Metric-normalization subtlety. The heatmap is min-max normalized to [0,1] before scoring (attribution.py:104-106). regions_above_threshold(0.5) is therefore relative to each image's own peak, and subtracting the min before topk_mass_ratio removes a uniform floor — both make the metric size- and shape-sensitive in ways worth documenting.

6. Inherits Chapter 4's ranking bias. The attributed phrase is whichever one produced the top-ranked region — which for rule_1 is often the worker, not the hard hat (the by-phrase table lists "worker" as its own 0.115 row). So which entity gets its sparsity measured carries forward the area-ranking confound.

What scale-up needs
Two decisions and one metric fix:

Decision 1 (attribution method), the open question for Prof. Abdallah: is cross-attention acceptable evidence for a fused-modality VLM, or does a transformer-native method (LRP / DeepLIFT / integrated gradients — Abdallah's own white-box comparison set) need adapting first? This determines whether the paper claims all six components ported or five of six.
A size-normalized sparsity metric, so cross-rule comparison is valid at all.
Hindrances to scaling, and how to solve each
Hindrance	Why it bites at scale	Fix
Size confound (−0.70)	Large-object rules (excavator, guardrail) look artificially worse; any by-rule sparsity ranking is misleading	Replace/augment top-k-of-576 with a size-invariant measure — e.g. attention mass inside the grounded box vs cells in the box, or top-k normalized to expected-k-given-area
Attribution method not validated (decision 1 open)	Can't claim the attribution component is "ported"; reviewer may reject cross-attention as evidence	Adapt and verify a transformer-native method (LRP/DeepLIFT/IG) with the same architecture check rollout got; report cross-attention as a documented choice, not the only option
Greedy vs beam mismatch (verified on n=1)	Attributing the wrong decode at scale undermines every heatmap's link to the reported answer	Verify greedy≈beam boxes on a large sample, or implement the beam-attention un-reordering so you attribute the actual beam answer
Over-averaging diffuses the map	0.108 top-5 mass may under-report real focus across 3,004 images	Head/layer selection (informative heads, or last decoder layer) and/or per-token attribution instead of averaging across all generated tokens
Normalization choice affects the metric	min-subtraction + per-image max make numbers hard to compare across the full set	Fix and document one convention (e.g. mass ratio on raw softmax attention; threshold count on normalized) and keep it consistent
Attributed-phrase inherits Ch4 bias	Sparsity sometimes measured on the worker, not the safety object	After the rule-aware ranking fix, attribute the object phrase consistently; report by object class, not by whichever box was biggest
Off-by-one / grid arithmetic hardcoded	Breaks silently if resolution or patch grid changes	Keep the drop-global-token step behind an assertion; recompute 768→24 arithmetic rather than hardcoding 576
Grid-fallback exclusions (4/163)	Same rule_3 concentration as Ch4/Ch10	Improve rule_3 grounding prompt; fewer no-phrase exclusions
Compute is not the blocker (Chapter 8: cross-attention is one greedy forward pass per sample). As with 4 and 5, every issue here is validity — and the size confound is the one that most directly distorts headline numbers.

Through-line so far: Chapter 4's area-ranking still echoes here (which phrase gets attributed), and this chapter adds a second measurement confound — physical size — that, like the fallback-rerouting mechanism, is a property of the evaluation setup, not of Florence-2. The paper's methods section is accumulating a clear pattern: each metric needs an explicit control for an artifact the tabular original never had to worry about.

Chapter 7
Analysis for Chapter 7 (Stability). This chapter handles its main trap well up front, but its central finding is the third appearance of the same confound family — and it hides a subtler selection-bias issue in the code.

The trap it handled well
Stability means "rerun the same input, does the answer/region stay put?" But Florence-2's default decoding is deterministic beam search (num_beams=3), so literal reruns would report 100% stable by construction — a vacuous result. The fix: rerun 3× with stochastic sampling (do_sample=True, num_beams=1, temperature=0.7) and confirm non-vacuousness before trusting anything (answer_agreement_rate = 0.775 ≠ 1.0). That's the right instinct.

But note what it costs: the stability number now measures a decoding regime the pipeline never actually uses. Every "official" answer in the pilot is deterministic beam search; stability is measured under artificial temperature-0.7 sampling. So 0.775 isn't "how unstable is the deployed system" (that's 100% stable, trivially) — it's "how sensitive is Florence-2 to sampling noise at one arbitrary temperature."

Problems this chapter surfaced
1. The central finding: top_region_overlap_score (0.667) hides how unstable the safety object is — Chapter 4 again. Because top-1 is the worker box (area-ranking), the headline overlap mostly measures worker-detection stability. They added object_region_overlap_score on object_boxes[0], which exposes the gap: rule_1's hard-hat box overlaps only 0.499 vs rule_4's excavator at 0.922 — nearly 2×. On image 23 the worker boxes overlay almost perfectly (0.667-flattering) while the hard-hat box the rule is actually about scores 0.069 — a near-total miss sitting under a healthy-looking top-1 number.

2. Size + shape confound — the same mechanical artifact as Chapter 6, in a different metric. Small objects are least stable (hard hat 570 px² → 0.499; harness 7,169 → 0.533); large are most stable (guardrail 221k → 0.615; excavator 215k → 0.922). And it's not just size: guardrail and excavator have near-identical area (within 3%) but 0.615 vs 0.922 — a shape/rigidity effect (long, segmented guardrail vs compact, rigid excavator). The root cause is that IoU is inherently size-biased: a few pixels of edge jitter destroys IoU for a small box but barely dents a large one.

3. A selection bias hidden in the code. _object_region_overlap (07_stability_test.py:38-42) returns nan unless all three reruns detected an object box. So samples where the object box appears and disappears across reruns — the most unstable cases of all — are silently dropped, not scored as unstable. This means object_region_overlap_score is computed only on samples where the object was reliably present, biasing it upward. Detection-presence instability isn't captured anywhere.

4. Three decoding regimes across the pilot. Answers use beam-3, attribution (Ch6) uses greedy, stability uses sample+temp0.7. Each is individually justified, but at scale this fragmentation makes "what does the model do?" hard to reason about, and stability conclusions don't necessarily transfer to the deterministic regime everyone else's numbers come from.

5. n=3 reruns is coarse. Pairwise agreement over 3 reruns can only be 0, 0.33, 0.67, or 1.0 per sample; the majority vote is a 3-way coarse signal. Fine for a pilot, noisy per-sample.

6. The stability number is temperature-dependent and arbitrary. 0.775 is a function of the chosen temperature=0.7. A different temperature gives a different headline with no principled anchor.

They did correctly rule out box reordering as an alternative explanation (only 4–8% of samples have >1 object box), so the instability is real box movement — a solid check.

What scale-up needs
Report both overlap metrics by rule (the chapter's own general recommendation): the heuristic's pick and the metric's true target object, never the pick alone.
A size-invariant stability measure so the geometry confound doesn't masquerade as instability.
An object-presence-stability metric so disappearing boxes count as instability instead of vanishing into nan.
Hindrances to scaling, and how to solve each
Hindrance	Why it bites at scale	Fix
IoU size bias / small-PPE penalty	hard hat & harness score worst for geometric reasons, not explanation quality — exactly the PPE items trust matters most for	Complement IoU with size-invariant measures: normalized center-distance, or Dice; report geometry-controlled stability
nan-dropping the hardest cases	Object stability is biased upward; worst instability (box vanishes) is invisible	Add an object-presence rate (fraction of reruns detecting the object) as a separate stability signal; don't exclude disappearing boxes
top_region_overlap dominated by the worker	Headline understates safety-object instability (Ch4 inheritance)	Rule-aware ranking (Ch4 fix); always pair the two overlap scores
Temperature-dependent, arbitrary number	0.775 has no principled anchor; not comparable across studies	Sweep temperature and report a stability-vs-temperature curve, or tie temperature to intended deployment; frame the deterministic system honestly as "stable by construction, sensitivity measured under sampling"
n=3 reruns coarse	Noisy per-sample estimates on 3,004 images	Increase reruns (5–10); compute is cheap (Ch8)
Three decoding regimes	Stability may not transfer to the beam-search regime the answers use	Document each metric's regime; cross-check via input-perturbation stability (deterministic decoding, perturbed input) — which is exactly what Ch9 robustness does
Small-n class inflation	rule_4's 0.922 is partly n=25/struck-by-13	Decision 2: minimum n before a rate is called stable
Compute is not the blocker (Ch8): stability is 3 forward passes/sample; even n=10 reruns plus a temperature sweep stays cheap.

Through-line, now three chapters deep: the area-ranking (Ch4) and the object-size confound (Ch6) both reappear here, and this chapter contributes the general recommendation that ties them together — report the heuristic's pick and the metric's true target separately. Chapter 11's cross-metric analysis is where Ch6's size confound and Ch7's size+shape confound should be consolidated, so rule_1/rule_2's weak numbers aren't misread as "worse explanations for PPE" when they're substantially geometry.

Want me to prototype the two validity fixes — an object-presence-stability metric and a size-invariant center-distance stability score — and recompute the by-rule table to see how much of the 0.499-vs-0.922 gap is geometry versus real instability? Behind a flag, so the recorded pilot numbers stay put.


Chapter 10
Chapter 10: Bounded Completeness: Is the Top Region Actually Necessary?
In Professor Abdallah's framework, "Completeness" evaluates whether an XAI method can provide a mathematically valid explanation for every possible sample, including rare corner cases
. Your pilot tests this by observing whether removing the top-ranked region actually forces the VLM to change its prediction
.
Your execution in this chapter reveals an exceptional level of technical rigor—specifically your dedication to running an extra 159-sample test just to double-check your own rollup
. However, this chapter also serves as the final, fatal indictment of your area-ranking heuristic.
Here are the three critical methodological gaps you must address before scaling this metric up to your cluster run:
1. The Fatal Flaw: The Area-Ranking Bias Ruins the Metric
The Issue: Your headline result states that 54.0% of the explanations were "weak"—meaning you masked the top region, but the model's answer didn't change
. However, your Finding 1 proves this failure is heavily skewed by the area-ranking heuristic. Masking the safety object successfully flipped the answer 49.4% of the time, but masking the worker flipped it only 37.5% of the time
.
Why Professor Abdallah will reject it: Professor Abdallah’s white-box methods (like LRP or DeepLIFT) calculate true mathematical importance scores directly from the neural network's weights
. Because Florence-2 lacks this native output, you used bounding-box area as a stand-in, which artificially promoted the worker to "Top-1" in 96% of Rule 1 (PPE) cases simply because workers are bigger than hardhats
. Your 54.0% failure rate doesn't prove the VLM's explanations are incomplete; it proves your Python script is systematically testing the wrong bounding box.
The Fix: As you noted in your implications, scaling to the cluster requires abandoning the area-based ranking
. You must implement a "rule-aware" semantic ranking that prioritizes the queried object class (e.g., matching the "hard hat" box first) before falling back to area
.
2. The Broken Promise of "Corner Cases"
The Issue: In your original E-XAI framework pitch to Professor Abdallah, you explicitly promised that your Completeness metric would test difficult edge cases, specifically "overlapping worker-equipment interactions" and "multiple hazards in one image"
.
The Flaw: In your Table 10.2 breakdown, you admit that you only had n = 7 multi-rule-violation samples, which is statistically too small to draw any conclusions from
. By failing to test multi-hazard scenarios, you have failed to evaluate the core definition of "Completeness" as established in Professor Abdallah's cybersecurity literature, which demands verifying explanations for complex outliers and corner cases
.
The Fix: When you move to the full 3,004-image test split on the cluster, you must deliberately stratify and oversample multi-hazard images. You need to prove that if a worker is missing a hardhat and standing near an unprotected edge, the VLM's explanation is complete enough to capture both.
3. The "Worker_Lost" Artifact Inflates the Results
The Issue: You brilliantly discovered in Finding 2 that your "explanation_supported" metric was artificially inflated. In 3 out of your 71 "successful" cases (4.2%), masking the top region didn't actually change the model's visual reasoning; it simply erased the only worker in the frame
. This caused the VLM to return zero worker boxes, which crashed into your Python script's scene-level fallback branch and blindly flipped the answer
.
The Fix: You correctly adjusted your true "supported" count down from 43.6% to 41.7%
. For the final paper, you must hardcode a filter that completely strips any worker_lost=True samples from your Completeness and Robustness scores, explicitly stating that masking operations that break the detection pipeline cannot be counted as valid XAI evaluation
.
Summary for Chapter 10: Your rigorous self-correction regarding the worker_lost artifact is exactly the kind of intellectual honesty top-tier journals look for
. However, this chapter proves definitively that the area-ranking heuristic completely bottlenecks your ability to evaluate PPE rules
.

I've now read all five remaining chapters (10–14). Their character shifts: Chapter 10 is the last real metric, while 11–14 are synthesis/process — so the issues move from measurement confounds to aggregation discipline, number-staleness, and environment reproducibility. I'll do each compactly, then give a consolidated scaling checklist across the whole pilot since this is the last batch.

Chapter 10 — Bounded Completeness
What it does: necessity test — is the top region load-bearing? Rolls up Ch5's answer_changed columns + Ch4's top_region_source. Headline: 43.6% supported, 54.0% weak, 2.5% no-usable.

Problems:

It's not an independent metric. classify_sample() is built from Ch5's answer-change columns by construction — Ch11 Finding 1 confirms it's "close to a methodological identity." So completeness and descriptive accuracy are one finding, not two.
Bounded scope: only top-1 / top-1+2 ablation, never a full completeness sweep — a region necessary only in combination with others is invisible.
Area-ranking bias (Finding 1): the pilot's single highest-leverage flaw, measured here — masking the worker flips 37.5% vs 49.4% for the real object; rule_1 weakest at 33.3%.
Worker-fallback artifact (Finding 2): inflated the headline 71→68/163. Critically, detecting it required an ad-hoc extra 159-rerun set — the metric itself doesn't catch it.
Confidence proxy ruled out as a passing signal (non-discriminative), so the verdict rests on the coarse binary answer alone.
Scale-up fixes: bake the worker-loss correction into the metric (don't leave it as an extra pass); fix Ch4 ranking; report genuine vs raw supported rate; consider a deeper (not just top-2) ablation; minimum-n before reporting.

Chapter 11 — Cross-metric rollup
What it does: groupby(primary_class).mean() over the six metrics — one summary table + chart. No new computation.

Problems:

Double-counting risk (Finding 1): descriptive accuracy and completeness share a shape because one is built from the other — not independent corroboration.
n=13 pollutes the whole table. Every struck_by_risk cell rests on 13 samples, yet struck_by is the "strongest class" for DA, stability, and completeness. Those "best-class" claims are directional only.
Class aggregates blend heterogeneous mechanisms (Finding 3): fall_hazard pools rule_2 (fallback fragility) and rule_3 (grounding difficulty) — the class number hides both causes.
The rollup groups by primary_class, but the metrics were computed per assigned_rule; pooling rules into classes mixes structurally different failure modes.
Scale-up fixes: always report sub-rule breakdowns alongside class-level; fix small-n; explicitly annotate which metric pairs share construction so agreement isn't read as independent evidence.

Chapter 12 — Pilot report memo
Process chapter, no new numbers. Its real contribution is naming the two recurring mechanisms (fallback-rerouting; area-ranking) and a strict "every number traces to a CSV" discipline.

The one scaling hindrance: that discipline is manual. Numbers are hand-copied from CSVs into prose. At scale — with corrected metrics, larger n, and iterating results — this guarantees drift: the memo goes stale the moment a metric is recomputed. This is the same class of defect Ch13 then finds in hardcoded chart labels.

Scale-up fix: generate report numbers programmatically (pull from results CSVs at build time, e.g. templated Markdown/LaTeX with \input of computed values) so the writeup can never disagree with the data.

Chapter 13 — Reproducibility (the most scaling-relevant of the five)
What it does: clean-room 20-sample rerun from a fresh git archive + fresh venv. All 12 steps exit 0.

What it surfaced — both directly relevant to scaling:

A real hardcoded-assumption bug: 11_make_figures.py hardcoded "163-sample pilot" and "13 if struck_by_risk else 50" legends. Invisible for 12 days because every prior run was n=163. Fixed to read from data. The lesson generalizes: any other hardcoded n/class-size/magic-number assumption breaks the moment you change n — which is exactly what scaling does.
Transient streaming failure (IncompleteRead) auto-recovered by HF's 3-retry budget. At full-dataset scale or worse networks, >3 consecutive failures fails the whole run. This directly validates the local-download work we already did — reading local parquet removes this failure mode entirely.
20-sample numbers diverge (stability 93.3% vs 77.5%) — sampling noise; ordering reproduces, exact numbers don't. Ties to the feedback doc's single-SEED=42 concern (decision 2).
Fragile install order: a generic requirements.txt torch entry can silently replace the CUDA build with CPU-only. Real environment fragility at cluster scale.
Scale-up fixes: audit every script for hardcoded-n/class assumptions before scaling; containerize the environment (or pin + document install order); keep the run local/GPU-bound (done) or configure --retries/timeouts; run multiple seeds at scale, not just SEED=42.

Chapter 14 — Meeting package
Packaging chapter. Its substance is converting open items into six decisions — which are the scale-up roadmap: (1) attribution method, (2) small-n/struck_by statistics, (3) region-ranking fix, (4) full-scale vs fix-imbalance, (5) powered Level-3 robustness, (6) rule_2 prompt split. The only "hindrance" is that scaling can't proceed cleanly until these are decided; the package correctly hands them to Prof. Abdallah with evidence attached.

Consolidated scaling checklist (Chapters 4–14)
Priority-ordered, since this is the last batch. The feedback doc you keep opening maps almost one-to-one onto these.

#	Issue	Origin chapters	Fix	Leverage
1	Area-ranking picks the worker, not the PPE object	4, 6, 7, 10, 11	Rule-aware ranking (match queried class before area)	Highest — improves 4 metrics at once, no new model calls (decision 3)
2	Fallback-rerouting inflates/mis-attributes flips	5, 9, 10	Instrument post-mask worker_boxes; report genuine vs raw	High — corrects 3 chapters' headline numbers
3	Object-size confound (IoU / top-k mechanics)	6, 7, 9(Finding 2)	Size-invariant metrics (mass-in-box, center-distance); verify Ch9 centerpiece survives	High — may de-risk the pilot's best result
4	Small-n, especially struck_by (n=13)	2, 11, 13	Minimum-n rule; multi-seed; mine more struck_by (decisions 2, 4)	High
5	Attribution method not validated	6	Adapt+verify LRP/DeepLIFT/IG, or accept cross-attention explicitly; cost it (decision 1)	Medium — determines "6/6 vs 5/6 ported"
6	Hardcoded assumptions break at new n	13	Audit all scripts; parametrize; programmatic report numbers	Medium — silent-wrongness risk
7	Single-point perturbation/temperature magnitudes	7, 9	Severity/temperature sweeps (dose-response)	Medium
8	Streaming I/O fragility	8, 13	Local caching — done; else configure retries	Resolved
9	rule_2 phrasings aren't synonyms; rule_3 grounding weak	9, 10, 11	Split rule_2 into two concepts; better guardrail prompt (decision 6)	Medium
10	Report staleness (manual number-copying)	12	Template reports from CSVs	Low-medium
Two reassurances that hold across all of it: efficiency is genuinely not a blocker (Ch8, ~0.5 GPU-hr/1000), and the pipeline reproduces from scratch (Ch13). Every item above is about validity and maintainability, not feasibility — and items 1–3 share roots, so a single push (rule-aware ranking + worker-loss instrumentation + size-invariant metrics) de-risks Chapters 4, 5, 6, 7, 9, and 10 together.

Want me to turn this consolidated checklist into a standalone document — either a Markdown "scaling readiness" memo in pilot/report/, or a shareable artifact — so it sits alongside the meeting package? Or start prototyping item 1 (rule-aware ranking), which is the single highest-leverage change and unblocks the most downstream fixes?