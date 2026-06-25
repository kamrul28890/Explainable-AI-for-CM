I have scrutinized Chapter 1 with that scale-up in mind. There are several technical and methodical issues hidden in these early decisions that will cause friction or compromise your research rigor when you scale up.
Here is a detailed breakdown of the issues in Chapter 1:
1. Technical & Scaling Issue: The "Streaming Mode" Bottleneck
The Issue: In Section 1.2 and 1.3, you establish that the pipeline reads the ConstructionSite 10k dataset from the Hugging Face Hub in streaming mode
.
Why it's a problem: While streaming is great for a 163-sample local Windows pilot, it is highly unsuitable for running a huge dataset on a computing cluster. Streaming relies entirely on continuous, uninterrupted internet I/O. If you run thousands of images across multiple GPU nodes, you will bottleneck the cluster's GPUs while waiting for network downloads, and you are highly vulnerable to connection drops. In fact, your own report in Chapter 13 already catches a glimpse of this exact error: a transient IncompleteRead/ProtocolError network blip caused by Hugging Face streaming
.
The Fix: Before moving to the cluster, you must change your dataset loading code to download and cache the entire dataset locally on the cluster's high-speed storage (e.g., NVMe/SSD) to ensure GPU-bound, rather than I/O-bound, performance.
2. Methodical Issue: fp16 Precision vs. White-Box Gradients
The Issue: Section 1.3 and 1.4 state that Florence-2 is running in fp16 (half-precision)
.
Why it's a problem: fp16 is standard for forward-pass inference to save memory, but Professor Abdallah's white-box XAI methods (Integrated Gradients, DeepLIFT, LRP) rely heavily on calculating internal neural network gradients
. fp16 is notoriously prone to gradient underflow/overflow (where small gradient numbers simply round to zero) compared to fp32 or bf16. If you extract gradients for your XAI metrics in fp16, your sparsity or completeness scores might be artificially skewed because precision was lost in the math, not because the model wasn't looking at the right features.
The Fix: When you run the XAI attribution extractions (especially if you later implement Transformer-native white-box methods), you should test if running the model in bf16 (bfloat16) or fp32 alters your XAI feature importance rankings.
3. Technical Debt: The transformers==4.49.0 Hard Pin
The Issue: To avoid a configuration crash caused by transformers>=5.0 removing legacy generation attributes, you forced a hard pin to transformers==4.49.0
.
Why it's a problem: When you move to a cluster, you will likely be using pre-configured modules or containerized environments (like Docker or Singularity) optimized for the cluster's specific NVIDIA A100 architectures (which Prof. Abdallah's lab uses)
. Tying your entire research stack to an outdated 4.x version of transformers risks dependency conflicts with the newer versions of PyTorch or CUDA required by cluster environments.
The Fix: As you noted in Section 1.7
, this is a fragile workaround. Instead of pinning an old version, a more robust research practice is to write a custom config-wrapper class in your model.py that intercepts the Florence-2 config load and manually injects the missing attributes (like forced_bos_token_id). This allows your code to survive upstream package updates.
4. Technical / Reproducibility Issue: sdpa Attention Differences Across Hardware
The Issue: To bypass the Windows flash_attn crash, you monkeypatched the code to force attn_implementation="sdpa" (Scaled Dot-Product Attention)
.
Why it's a problem: PyTorch's sdpa acts as a dispatcher. On your Windows machine, it is falling back to mathematical or memory-efficient attention because flash_attn isn't available. When you move to the Linux-based cluster with A100 GPUs, sdpa will likely detect the hardware and automatically route to flash_attn under the hood. Flash Attention is highly optimized but is known to be non-deterministic in its backward pass. This means the exact same image might yield slightly different attribution gradients on your Windows machine versus the cluster.
The Fix: If you want strict mathematical reproducibility for your XAI scores across platforms, you need to be aware that your Windows pilot and your cluster run might yield slightly different baseline gradient matrices due to this hardware-level attention routing.
5. Research Rigor: Static Seed Dependency
The Issue: You explicitly centralized SEED = 42 to keep everything deterministic
.
Why it's a problem: Setting a fixed seed is perfect for a pilot to ensure your pipeline runs reliably. However, when you write the final journal paper, relying on a single random seed for drawing your balanced dataset (as done in Chapter 2)
 can introduce selection bias.
The Fix: When you run the massive dataset on the cluster, ensure your methodology allows for cross-validation across multiple random seeds, rather than treating SEED = 42 as an absolute ground truth for your final paper's statistics.


Chapter 2

1. The "Struck-by" Scarcity is Likely an Artificial Bug, Not a Dataset Property
The Issue: Your report states that you could only find 13 struck_by_risk samples in the entire 3,004-image test split, forcing you to proceed with a heavily imbalanced dataset
. The report confidently states this is a "property of the dataset itself, not a limitation of the sampling method"
.
The Flaw: This scarcity is almost certainly caused by your CLASS_PRIORITY logic. In Section 2.2, you state that if a row violates multiple rules, it is assigned a single primary class based on the priority list: ["ppe_violation", "fall_hazard", "struck_by_risk", "compliant"]
. Because struck_by_risk is at the very bottom of the violation hierarchy, any image that contains a struck-by risk and a missing hard hat (PPE violation) is strictly labeled as a PPE violation, effectively erasing the struck-by data.
The Fix: For a top-tier journal, you cannot use mutually exclusive priority sorting for multi-hazard environments. Construction safety is multi-label by nature. When you move to the cluster for the full run, you must rewrite classify_image() to allow multi-label stratification, or you will artificially starve your minority classes.
2. The Statistical Invalidity of n = 13
The Issue: You moved forward with only 13 samples for the struck_by_risk class
.
Why Professor Abdallah will flag this: His evaluation frameworks rely heavily on rigorous statistical validation. In his prior work on IDS, he uses non-parametric Wilcoxon signed-rank statistical tests to calculate p-values and prove that his XAI framework's findings are statistically significant
. You cannot run meaningful statistical significance tests on XAI metrics (like Stability or Robustness) with a sample size of 13.
The Fix: As noted in your implications section, this is an open question for Professor Abdallah
. For the final paper, you will likely need to explicitly mine your other datasets (like the CMA dataset
) to supplement this specific class, ensuring you have a statistically valid sample size across all hazard categories.
3. The "Round-Robin" Compliant Sample Bias
The Issue: Because compliant images don't violate any rules, you used a non-standard "round-robin" assignment to distribute them evenly across testing rules (e.g., Rule 1: hard hat, Rule 2: harness)
.
The Flaw: If you randomly assign a "Rule 2 (Harness/Fall Protection)" query to a compliant image of a worker standing on flat ground, the VLM will answer "compliant." But this isn't a true verification of safety; the hazard context doesn't even exist in the image. This could artificially inflate your baseline proxy accuracy and severely skew your Descriptive Accuracy tests later on.
The Fix: You need to ensure that the compliant samples are contextually matched. If a compliant sample is assigned to Rule 2, the script must verify that the image actually contains a worker at height (the context for a harness).

1. Semantic Aliasing in RULE_QUERIES (The "Synonym" Trap)
The Issue: In Section 2.2, you establish the RULE_QUERIES that the Vision-Language Model (VLM) will use. For rule_2, you group ["safety harness", "fall-protection lanyard"] together as phrasings for the same rule
.
Why it's a problem: You treat these as wording variants or synonyms of a single concept. However, a safety harness is a body garment, while a lanyard is a tether attached to it—they are two distinct physical objects
. If you randomly swap these prompts, Florence-2 will accurately ground two entirely different bounding boxes in the same image
.
The Fix: Your report actually catches this later in Chapter 9 as a "prompt-design issue"
, but the root cause is here in Chapter 2. Before scaling up to the cluster, prompts.py must be rewritten. Do not group distinct physical objects as "phrasings" of the same rule. Each distinct object requires its own strict semantic prompt to ensure the XAI is evaluating the correct visual pixels.
2. Designing Out the "Completeness" Corner Cases
The Issue: In your original research proposal to Professor Abdallah, you explicitly promised that your "Completeness" metric would test complex corner cases, specifically emphasizing "multiple hazards in one image"
.
Why it's a problem: In Section 2.2, your assign_rule_id logic explicitly forces the assignment of only one rule per sample
. Even if a row violates multiple rules (e.g., a worker is missing a hardhat and near an edge), your script only assigns the one rule that matches the primary_class
. By artificially restricting every test sample to a single safety rule, you have structurally designed out the multi-hazard corner cases needed to truly evaluate the Completeness metric
.
The Fix: To fulfill the promise of your proposal, your data preparation script must be updated to allow a single image to be passed to the VLM with multiple concurrent safety queries. You need to test if the VLM (and its XAI explanation) gets confused when asked to identify a missing hardhat in a scene that also features an active struck-by hazard.

Chapter 3
1) Is 13 the true population, or an artifact?
It's an artifact — confirmed, not suspected. I scanned both splits directly against the live dataset, counting rows two ways: raw presence of rule_4_violation (regardless of other rules), vs. the priority-collapsed primary_class that config.py:25's CLASS_PRIORITY = ["ppe_violation", "fall_hazard", "struck_by_risk", "compliant"] actually produces:

Split	Rows with rule_4_violation set (raw)	Rule_4-only (what got reported)	Absorbed into a higher-priority class
test (3,004 img)	24	13	11 (10→ppe_violation, 1→fall_hazard)
train (7,009 img)	46	34	12 (11→ppe_violation, 1→fall_hazard)
combined (10,013 img)	70	47	23
So: the day02 chapter's claim that 13 is confirmed by "an exhaustive scan" is technically accurate but incomplete — it exhaustively scanned the test split after the priority collapse, not the raw annotation rate, and not the full dataset. The actual mechanism, as feedbacks-for-full-project.md correctly flags: classify_image (data.py:23-41) assigns one mutually-exclusive class per row, and struck_by_risk sits last in the priority list — so any image that has both a struck-by hazard and a PPE/fall violation gets silently relabeled into the other class. 11 of the 24 true struck-by-risk test-split images were thrown away this way.

To your literal question — could we have pulled 50 from the full 10k? With the current mutually-exclusive scheme, no: even pooling train+test only gets you to 47 (still short of 50). But that's an artificially low ceiling. If classify_image instead allowed multi-label assignment (an image can be both ppe_violation and struck_by_risk), the real pool is 70 images across the full dataset — comfortably past 50. This is exactly the fix your feedback doc proposes, and the numbers above confirm it's not a hypothetical concern — it measurably cost you 11 of 24 real examples in the split you actually used.

This is worth fixing before scaling, not something to defer — and it's a quick fix: classify_image needs to return the full set of violated classes (it already computes violated_classes internally) rather than collapsing to one primary_class, and select_balanced_sample needs to sample with a multi-label-aware criterion instead of bucketing by a single winner.

2) Is Florence-2 even comparable to a native-VQA model?
You're right to be suspicious here, and the honest answer is: not directly, without an explicit caveat in the methods. The reason is structural, and it's visible in the code, not just in principle.

Look at what's actually producing the "answer" in the Florence-2 pipeline. _answer_presence_rule (inference.py:81-119) calls Florence-2 only to get boxes — "worker" boxes and "hard hat" boxes. The compliant/violation judgment itself is computed by our own Python code, not the model:


covered = all_boxes_covered(worker_boxes, object_boxes, distance_threshold=distance_threshold)
answer = "compliant" if covered else "violation"
all_boxes_covered (regions.py:94-105) is a hand-written geometric rule — IoU overlap or a hard-coded distance threshold (PRESENCE_PROXIMITY_FRACTION = {"rule_1": 0.08, "rule_2": 0.08, "rule_3": 0.20} in inference.py:41-45). Florence-2 never makes the safety judgment; it only locates objects, and we decide compliance from coordinates.

A native-VQA model like Qwen2.5-VL, asked "is the worker wearing a hard hat?", makes that judgment inside the model — using whatever multimodal reasoning it learned (posture, occlusion, context), not a fixed pixel-distance rule.

This matters a lot for your six XAI metrics specifically, not just for raw accuracy:

Descriptive accuracy / completeness (does masking the top region flip the answer) is almost tautological for Florence-2's proxy — if you black out the hard-hat box, object_boxes becomes empty, all_boxes_covered mechanically returns False, and the answer flips by construction. That's measuring your threshold logic's sensitivity to its own input, not Florence-2's "faithfulness." For Qwen2.5-VL, the same masking test measures something genuinely different — whether the model's judgment depended on that pixel region.
Stability/robustness numbers are similarly comparing different things: Florence-2's stability reflects detection-box jitter run through a fixed rule; Qwen2.5-VL's stability reflects the LLM decoder's own answer variance.
So if you ran both models through this evaluation and reported the same six numbers side by side, a reviewer who looks closely (and at a 12+ IF venue, someone will) could correctly object that you're not comparing two VLMs' explanation quality — you're comparing "a detector plus a hand-tuned rule" against "an end-to-end model's own judgment." That's a confound, not a controlled comparison.

The fix, concretely: run two comparisons, not one, and label them as different things:

Same-proxy comparison — also constrain Qwen2.5-VL to the identical detect-then-threshold pipeline (it supports open-vocabulary grounding too), so the only thing that varies is detector quality. This isolates "does a better detector make the existing rule's explanations more faithful."
Native-VQA comparison — separately, let Qwen2.5-VL answer the rule directly as free-text VQA, with its own attention/gradient-based attribution. This isolates "does end-to-end reasoning produce better-quality explanations than a geometric proxy" — a genuinely different question.