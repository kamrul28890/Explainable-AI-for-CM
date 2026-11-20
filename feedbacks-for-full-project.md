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

Chapter 3
1. The "Single Worker" Detection Bottleneck Invalidates Completeness
The Issue: In Section 3.8, you explicitly note that querying Florence-2's <OPEN_VOCABULARY_DETECTION> with the phrase "worker" only returns one bounding box per scene, even if multiple workers are present
. You chose to leave this as a "bounded limitation" rather than fixing it with <DENSE_REGION_CAPTION> to avoid conflating variables
.
Why Professor Abdallah will flag it: A safety inspector that only looks at 1 out of 5 workers in a frame is inherently unsafe. More importantly, this breaks the "Completeness" metric you promised him. Completeness requires testing edge cases like "multiple hazards in one image"
. If your baseline pipeline literally cannot detect more than one worker, it is mathematically impossible to evaluate the XAI's completeness in multi-worker hazard scenarios.
The Fix: You cannot defer this to future work. Before running the full dataset on the cluster, you must swap the task token to one that guarantees exhaustive instance enumeration (like <DENSE_REGION_CAPTION>) so that every worker in the frame is evaluated.
2. Hardcoded 2D Proximity Thresholds Regress Back to Legacy CV
The Issue: To determine if a worker is wearing PPE (rules 1 and 2) or protected by a guardrail (rule 3), your v2 proxy checks if the safety object's bounding box is within a hardcoded 2D pixel distance from the worker's box: 8% of the image dimension for PPE, and 20% for guardrails
.
Why it weakens your paper's thesis: In your literature review, you explicitly state that traditional Computer Vision fails because it relies on "manual geometric mapping"
 and lacks semantic reasoning
. By hardcoding 8% and 20% pixel-distance thresholds to decide if a worker is "wearing" a harness, you are reverting directly to the manual geometric mapping you criticized. These 2D thresholds will fail catastrophically due to camera perspective (e.g., a hardhat in the foreground appearing to overlap a worker in the background).
The Fix: Instead of hardcoded pixel thresholds, you should leverage the VLM's native spatial reasoning. Use a relational prompt (e.g., <CAPTION_TO_PHRASE_GROUNDING> with "worker wearing a hardhat") rather than detecting them separately and writing Python code to calculate if their boxes touch.
3. The Rule 4 (Struck-By) False Positive Trap
The Issue: You claim Rule 4 (proximity to an excavator) was unaffected by the v1 failure and showed strong discriminative signal (61.5% sensitivity) because it does a relational check
.
The Flaw: Because of the "single worker" bug mentioned above, Rule 4 is highly exposed to false negatives. If an excavator is swinging dangerously close to Worker A, but Florence-2 randomly selects Worker B (who is standing 50 feet away in the safe zone) as the only "worker" bounding box to return
, your Python script will calculate the distance between Worker B and the excavator and falsely classify the scene as safe.

Chapter 4 & 5
1. The "Area-Ranking" Heuristic Destroys PPE Evaluation (Chapter 4)
The Issue: Because Florence-2 does not natively output a true saliency or attention map during grounding, you made the decision in Chapter 4 to rank the candidate explanation boxes purely by pixel area, descending
. You used this as a stand-in for "importance"
.
Why Professor Abdallah will reject it: Construction safety heavily involves small objects (like hard hats or harnesses) attached to large objects (like workers). By ranking importance based on area, your script systematically promotes the worker's whole-body bounding box to "Top-1" and downgrades the actual PPE object
. When you test Descriptive Accuracy by masking the "Top-1" region, you are almost always just blacking out the worker, not the safety equipment the model was actually queried about
. This means your Descriptive Accuracy score completely fails to evaluate if the VLM understands PPE.
The Fix: You cannot use box area as a substitute for XAI importance. As you noted in your own implications, you must replace the area-only ranking with a rule-aware ranking
. The script must rank regions based on whether they semantically match the queried object class (e.g., matching the "hard hat" bounding box first), and only fall back to area if no semantic match exists
.
2. The Non-Monotonicity Anomaly: Testing Python, Not the VLM (Chapter 5)
The Issue: In Chapter 5, you found a bizarre anomaly: masking both the Top-1 and Top-2 regions resulted in a lower flip rate (35.0%) than masking just the Top-1 region alone (36.2%)
. You traced this to 15 cases where masking the second region actually "un-flipped" the answer back to its baseline
.
The Flaw: You brilliantly diagnosed why this happens: because the worker is ranked Top-1 (due to the area heuristic), masking the worker causes Florence-2 to find zero workers. Your Python script then triggers the if not worker_boxes fallback branch you wrote in Chapter 3, which defaults to a naive scene-level check
.
Why this ruins the metric: Professor Abdallah’s Descriptive Accuracy metric is designed to test the internal neural pathways of the AI model. Because your masking procedure accidentally triggers a hardcoded Python if/else fallback, your 36.2% score is completely contaminated. You are no longer evaluating the VLM's explanation; you are evaluating the side-effects of your own fallback routing
.
The Fix: You must exclude cases where the mask artificially triggers the worker_lost fallback from your final Descriptive Accuracy metric
. Alternatively, you need to transition to true Transformer-native attribution methods (like the Cross-Attention you explore in Chapter 6) to find the important pixels, rather than relying on bounding-box deletion that breaks your own code's logic.
3. Missing the Semantic/Textual Half of the Multimodal Tuple
The Issue: In your E-XAI mapping document for Professor Abdallah, you explicitly defined your XAI units as "multimodal feature tuples" (visual regions + semantic/textual safety prompts)
. You promised to test Descriptive Accuracy by masking visual regions AND removing key regulatory prompt tokens
.
The Flaw: Chapter 4 and Chapter 5 only implement bounding-box masking on the image
. You completely dropped the semantic half of the test. You did not test what happens to the VLM's accuracy when you remove words like "hard hat" or "harness" from the query.
The Fix: To publish in a top journal and fulfill your proposal, your Descriptive Accuracy pipeline must be updated to systematically mask the most important text tokens in your prompt, alongside the visual regions.


Chapter 6
hapter 6: Visual Sparsity via Decoder-Encoder Cross-Attention.
This chapter is arguably the most technically impressive part of your report, but it also introduces a severe methodological inconsistency that Professor Abdallah will immediately flag.
In this chapter, your goal was to measure Visual Sparsity (how concentrated the VLM's attention is when making a safety judgment)
.
Here is my rigorous breakdown of your brilliant catch, followed by the two critical technical flaws you must fix before scaling up to the cluster:
The Massive Win: Catching the "Attention Rollout" Trap
The Brilliance: Your original plan was to use textbook "Attention Rollout." However, you looked at Florence-2's source code and realized that its encoder fuses image patches and text tokens into a single shared self-attention stack (1 global token + 576 image patches + text prompt)
.
Why Professor Abdallah will love this: If you had blindly applied Attention Rollout, you would have been recursively multiplying attention matrices that mixed image and text modalities together, resulting in mathematically invalid pixel attributions
. By catching this and substituting Decoder-Encoder Cross-Attention, you saved the project from a fatal mathematical flaw
. This demonstrates exactly the kind of deep architectural awareness Professor Abdallah expects.
However, your implementation of this cross-attention extraction introduces two major vulnerabilities for a top-tier journal submission:
1. The Methodological Flaw: The "Greedy Decoding" Mismatch
The Issue: In Chapters 3, 4, and 5, your baseline predictions and masking tests were all generated using Beam Search (num_beams=3)
. But when you extracted the cross-attention in Chapter 6, you silently switched the model to Greedy Decoding (num_beams=1) because Hugging Face's beam search scrambles the attention batch dimensions
.
Why it ruins the rigor: You are evaluating the XAI (explainability) of a completely different inference path than the one that actually made the baseline safety judgment. Even though you did a "spot check" on Image 7 and claimed the greedy boxes matched the beam-search boxes closely
, Professor Abdallah will point out that white-box XAI is strictly about faithfulness to the model's actual decision pathway
. You cannot decouple the decision-generation method from the explanation-generation method.
The Fix: Before moving to the cluster, you must unify your decoding strategy. Either (A) write the complex bookkeeping code to un-reorder the beam search attentions so you can extract them properly, or (B) switch your entire pipeline (starting from Chapter 3's baseline) to Greedy Decoding (num_beams=1). Option B is highly recommended, as Chapter 8 already proves that the pipeline's latency is dominated by the vision encoder, not the beam search
.
2. The Unresolved Confound: Unnormalized Sparsity Scores
The Issue: You correctly identified a massive "Size Confound." A small hard hat scores highly on sparsity (0.195) simply because it takes up very few pixels, while a massive excavator scores poorly (0.079) because it takes up a large portion of the frame
. You accurately noted that this makes direct comparison between safety rules impossible
.
The Flaw: While diagnosing this is great, stopping there is unacceptable for a top-tier publication. You cannot present a metric to a journal and simply state, "By the way, this metric is heavily skewed by the physical size of the object."
The Fix: You need to introduce a Normalized Sparsity Score. Instead of just measuring the raw concentration of attention, divide the attention mass area by the relative ground-truth bounding box area of the object in the image. If an excavator takes up 40% of the pixels, the model's attention should be spread over 40% of the pixels. Normalizing the sparsity score by object size will mathematically eliminate the confound and allow you to legitimately compare the explanation quality of a hard hat against an excavator.

Chapter 7
Chapter 7: Stability Under Stochastic Decoding.
This chapter evaluates the "Stability" metric, which Professor Abdallah defines as the XAI method's ability to consistently generate the same explanations over multiple identical runs
. Your handling of this chapter shows excellent experimental intuition, particularly in how you avoided a major trap. However, it also exposes a mathematical bias in how bounding boxes are evaluated that you must correct before the final paper.
Here is my rigorous breakdown of your brilliant catch, followed by the two critical technical flaws you need to address before scaling up to the cluster:
The Massive Win: Avoiding the "Vacuous Result" Trap
The Brilliance: You realized that running Florence-2 with its default deterministic beam search (num_beams=3) three times would yield a mathematically meaningless 100% stability score
. By intentionally switching to stochastic sampling (do_sample=True, temperature=0.7), you forced the model to explore its actual confidence distribution, revealing a genuine answer_agreement_rate of 77.5%
.
The "Hidden Instability" Catch: You also brilliantly diagnosed that looking at the top_region_overlap_score (0.667) masked the fact that the actual safety objects were highly unstable
. By explicitly computing a separate object_region_overlap_score, you proved that the hard hat bounding boxes were jumping wildly across runs (scoring a dismal 0.499 Intersection-over-Union)
. Professor Abdallah will highly appreciate this level of diagnostic depth.
However, your implementation of this stability test introduces two major vulnerabilities for a top-tier journal submission:
1. The Methodological Flaw: The IoU "Size Confound"
The Issue: You conclude in Section 7.4.2 that "object size and shape" drive the instability, noting that large excavators are stable (0.922 IoU) while small hard hats are unstable (0.499 IoU)
.
Why Professor Abdallah will flag it: Intersection-over-Union (IoU) is mathematically biased against small objects. If Florence-2 shifts a bounding box by 10 pixels on a massive 50,000-pixel excavator, the IoU barely changes. But if it shifts a bounding box by 10 pixels on an 800-pixel hard hat, the IoU plummets catastrophically. You are claiming that the VLM is "unstable" regarding PPE, but you are largely just measuring the mathematical denominator of the IoU formula.
The Fix: Just like the normalization fix required for Visual Sparsity in Chapter 6, you must introduce a Size-Normalized IoU or use a centroid-distance metric (e.g., measuring the pixel shift of the box's center point relative to the object's total size) rather than relying purely on raw IoU to claim that the model's explanations for PPE are unstable
.
2. The Structural Flaw: Decoding Mode Fragmentation
The Issue: Your pipeline is now completely fragmented across different decoding strategies.
Chapters 3, 4, & 5 (Baseline & Descriptive Accuracy) evaluate explanations using Beam Search (num_beams=3)
.
Chapter 6 (Visual Sparsity) evaluates explanations using Greedy Decoding (num_beams=1)
.
Chapter 7 (Stability) evaluates explanations using Stochastic Sampling (temperature=0.7)
.
Why it ruins the rigor: You cannot submit a paper to a top-tier journal where the "Explainable AI" metrics are computed on three entirely different neural network execution paths. Professor Abdallah’s E-XAI framework requires evaluating the explanation of the actual model prediction
. While you did compute that deterministic_matches_majority is 83.4%
, validating that the stochastic runs generally cluster around the baseline, evaluating the XAI metrics on fragmented decoding modes undermines the integrity of the framework.
The Fix: Before moving to the cluster, you must unify your pipeline. You should select one single decoding strategy (likely Greedy Decoding, as it solves your Chapter 6 cross-attention issue and is computationally efficient) and use it uniformly for baseline predictions, descriptive accuracy, sparsity, and the anchor for stability
.
3. The Area-Ranking Artifact (Again)
The Issue: Just like in Chapters 4 and 5, your top_region_overlap_score of 0.667 is artificially inflated because your Python script is ranking by pixel area, which almost always forces the worker's whole-body box to be "Top-1"
.
The Fix: This explicitly confirms what we discussed in earlier chapters: your area-based ranking heuristic is the single biggest bottleneck in this pilot
. When you scale to the cluster, you must transition to a semantic, rule-aware ranking, or your pipeline will continuously generate misleading metrics that evaluate the "worker" instead of the safety hazard
.

Chapter 8
Chapter 8: Efficiency and the Cost of Six Metrics at Scale.
This chapter evaluates the "Efficiency" metric, which Professor Abdallah explicitly uses to determine whether an XAI pipeline is computationally viable for real-world deployment
.
Your handling of this chapter demonstrates fantastic intellectual honesty. By openly admitting that you forgot to log the inference_ms in three of your scripts and documenting exactly how you recovered or estimated those numbers
, you show the kind of transparency expected in top-tier research. Furthermore, your Finding 1—realizing that you must normalize the "per-sample" cost by the actual number of internal generate() calls (which varied from 1 to 6 per sample) before comparing days
—is a brilliant piece of performance profiling.
However, looking at this through the lens of scaling up to a high-performance computing (HPC) cluster, your extrapolation methodology introduces three critical technical blind spots that will invalidate your efficiency claims if left uncorrected.
Here is what needs to be fixed before your final run:
1. The 15-Sample Extrapolation Fallacy
The Issue: To estimate the cost of the Cross-Attention (Day 6) and Stochastic Sampling (Day 7) methods, you ran a tiny 15-sample calibration and scaled it linearly to 163, and then extrapolated that up to 1,000 samples
.
Why Professor Abdallah will reject it: In his own cybersecurity evaluations, Professor Abdallah specifically measures efficiency at scales of 1,000, 2,500, and 10,000 samples
. He does this because deep learning attribution methods are highly prone to memory leaks and out-of-memory crashes over time
. A 15-sample burst completely masks GPU memory fragmentation, thermal throttling, and garbage collection overheads that only surface during sustained batch processing.
The Fix: As you rightly noted in your own implications section
, you cannot use calibration for the final paper. You must instrument the actual scripts for Days 6 and 7 to log real inference_ms across the entire dataset.
2. The Illusion of "Cheap" Beam Search (The Task Token Dependency)
The Issue: In Finding 2, you note that num_beams=3 (beam search) is surprisingly only ~4% slower than num_beams=1 (greedy sampling). You correctly deduce that this is because Florence-2's grounding task only generates 7–8 tokens, meaning the fixed cost of the vision encoder dominates the pipeline, making the decoder loop's beam search almost "free"
.
Why this is a ticking time bomb: Do you remember our fix from Chapter 3? To solve the "single worker" detection bug, we agreed you must switch your pipeline from simple object detection to exhaustive instance enumeration (like <DENSE_REGION_CAPTION>).
The Fix: Dense captioning will generate dozens or hundreds of tokens per image instead of 7-8. The moment you implement the Chapter 3 fix, the decoder loop will become the bottleneck, and the cost of num_beams=3 will exponentially skyrocket. Your efficiency estimates must be recalculated after you fix the detection task token, or your 0.50 GPU-hour estimate
 will be wildly inaccurate.
3. Ignoring I/O Overhead in the Efficiency Metric
The Issue: You claim the pipeline will cost roughly 0.50 GPU-hours for 1,000 samples
. However, you admit that when you ran a sanity check, the actual wall-clock time was much longer due to Hugging Face dataset streaming overhead
.
Why it breaks the metric: You are treating "Efficiency" purely as "GPU Compute Time." But in real-world construction management—especially on edge devices or site tablets—network latency and I/O are the primary bottlenecks. As we discussed in Chapter 1, streaming a dataset over the network causes transient protocol errors and stalling
.
The Fix: When you move to the cluster, your efficiency reporting must separate GPU Compute Time from End-to-End Wall Clock Time. You need to prove the pipeline is fast when data is cached locally on NVMe/SSD storage, rather than letting network streaming artificially inflate the perceived runtime of the model.

Chapter 8
 9: Robustness to Perturbation and Prompt Rewording.
This is arguably the most consequential chapter of your entire pilot. Your goal here was to test whether the Vision-Language Model's (VLM) judgments and explanations survive realistic physical perturbations (blur, lighting, occlusion, contrast) and prompt rewording
.
Your execution here yields a massive win for your paper's core thesis, but there are four methodological adjustments you must make before scaling this up to the cluster.
The Massive Win: Exposing the "Drifted Explanation"
Your Finding 2 is the crown jewel of this pilot. You discovered that in 28.3% of your stable-answer cases, the VLM's final safety answer did not change, but its underlying explanation drifted completely (scoring an Intersection-over-Union of less than 0.3)
. In your traced case of Image 34, occluding the worker kept the answer "compliant," but the VLM's evidence box jumped to a completely unrelated part of the image (IoU=0.005)
.
By officially separating answer-level robustness from explanation-level robustness into two never-blended numbers
, you have provided the exact, concrete proof Professor Abdallah needs to argue that standard VLM accuracy metrics are dangerously insufficient
.
However, looking at your setup for the cluster run, you must fix the following flaws:
1. Prompt Design: Stop Treating Distinct Objects as Synonyms
The Issue: In Finding 3, you note that your Level-2 rewording test caused a 50.0% answer-change rate for Rule 2 (harness vs. lanyard), compared to only 10.2% for Rule 3 (guardrail vs. edge barrier)
.
The Flaw: As you correctly diagnosed, a "safety harness" and a "fall-protection lanyard" are two distinct physical objects, whereas a "guardrail" and an "edge barrier" are true synonyms
. Testing linguistic robustness by swapping distinct physical objects is invalid because it legitimately forces the VLM to ground a completely different bounding box
.
The Fix: Before you run the full dataset, you must rewrite the RULE_QUERIES dictionary in your prompts.py file to ensure that your rewording test only swaps true linguistic synonyms
.
2. Drop the "Confidence Proxy" Entirely
The Issue: In Finding 4, you discovered that the VLM's mean confidence actually rose (a negative confidence drop) under occlusion and prompt rewording
.
Why it ruins the metric: These were your two most disruptive perturbations, yet the VLM grew more confident
. This definitively proves that Florence-2's reported confidence is an uncalibrated proxy measuring decoder commitment, not a true signal of correctness
.
The Fix: Do not report "confidence drop" as a metric for robustness in your final paper. It provides negative signal and will confuse reviewers
.
3. The Fallback Artifact Contaminates Robustness (Again)
The Issue: Just like we found in Chapter 5 with manual masking, your Finding 1 notes that 20% of the answer flips under occlusion occurred because the occlusion square happened to hide the worker entirely (worker_lost=True)
.
The Flaw: When the VLM loses the worker, your Python script triggers a degenerate scene-level fallback branch
. Your 28.2% occlusion flip rate is artificially inflated by your script's if/else logic firing, not by the model successfully noticing that safety equipment was obscured
.
The Fix: When computing your final Robustness scores on the cluster, you must systematically filter out or separately categorize any worker_lost=True cases
.
4. The Adversarial Stretch Test is Statistically Invalid
The Issue: You ran an adversarial "stretch test" by pasting a synthetic hardhat patch onto workers to see if it spoofed the model into predicting "compliant"
. You brilliantly caught your own denominator error: only 4 of the 20 samples were actually viable for the test, resulting in a 75% spoofing success rate (3 out of 4) rather than 15% (3 out of 20)
.
The Flaw: While your mathematical correction is perfect
, publishing a claim about a model's vulnerability to adversarial patches based on a sample size of n=4 is statistically invalid for a top-tier journal
. You also found no clean correlation between where the patch was placed and whether the model was fooled
.
The Fix: This is a great proof-of-concept, but you must scale up this Level-3 adversarial patch test significantly during your high-performance cluster run to generate a statistically powered result
.

Chapter 9
What it establishes
Aggregate per-sample inference time across the four inference-bearing days, extrapolate linearly to n=1,000: ~0.50 GPU-hours (~30 min) on a single RTX 3070 for the full six-metric pipeline. Efficiency is conclusively not the obstacle to scaling.

It also handles its instrumentation gap honestly (Day 3 was the only day that logged inference_ms; Day 5 recovered exactly from call-count × Day-3 timing, Days 6/7 got a 15-sample fresh calibration), and contributes two real findings: call-count normalization (a "call" bundles 2 generate calls for answer_rule vs 1 for attribution, so raw ms/sample isn't comparable) and beam-3 ≈ beam-1 here (outputs are only ~7–8 tokens, so fixed image-encoding cost dominates, not the decode loop).

Problems / weaknesses in the measurement
1. Days 6 & 7 rest on a 15-sample calibration, not the full run. The chapter flags this itself. Sanity-checked to order-of-magnitude (127 s implied vs. a few minutes observed), which is enough for "is it affordable," but not for precise per-sample numbers.

2. Inconsistent timing boundaries across days. Day 3/5/7 timings measure the generate portion (internal inference_ms), while Day 6's calibration uses wall-clock around the whole cross_attention_heatmap including preprocessing and post-processing (08_efficiency_test.py:92-94). Finding 1 normalizes for call count but not for this boundary mismatch — so Day 6's relative cost is measured against a different unit of work than it's compared to.

3. Warm-up contamination risk in a 15-sample calibration. The first CUDA calls trigger kernel compilation/allocation and run much slower; with only 15 samples and no discard-warm-up step, that skews Day 6/7's means upward. Related: time.perf_counter around async GPU ops is only correct if a sync happens before the stop (the .cpu() calls probably force it, but it isn't explicit).

4. Linear extrapolation ignores batching and throughput. extrapolate() is per_sample_ms × n — no batching, constant per-sample cost. A real full run would batch and get cheaper per sample, so the number is a conservative upper bound (fine), but it isn't a measured throughput and shouldn't be quoted as one.

5. Single hardware point, and the slow-attention path. The 0.5 GPU-hr is RTX-3070-specific. Day 6's attribution forces output_attentions=True → eager attention (65% more expensive per call), and how much that overhead costs differs across hardware — this is exactly the SDPA/FlashAttention routing issue in the feedback doc you have open. The number won't transfer unchanged to a cluster A100/H100.

6. The budget is for the current pipeline, not the corrected one. Every fix I've recommended in Chapters 4–7 adds inference: Ch10's 159 corrected reruns, Ch5's worker-loss instrumentation, Ch7's extra reruns and temperature sweep, and especially Ch6 decision 1 — if a transformer-native attribution (LRP / integrated gradients / DeepLIFT) is chosen, those need backward passes and multiple forward passes, potentially 10–50× a single forward pass. That cost is not in the 0.5 GPU-hr figure.

What scale-up needs
Instrument inference_ms directly into Days 6 & 7 (the chapter's own recommendation) and re-measure over the full run, retiring the 15-sample calibration.
Re-benchmark on the target (cluster) hardware with batching, reporting measured throughput rather than single-sample linear extrapolation.
Cost the corrected pipeline, especially the attribution-method decision, before committing.
Hindrances to scaling, and how to solve each
Hindrance	Why it matters at scale	Fix
15-sample calibration for Days 6/7	Precise per-sample numbers aren't trustworthy	Add inference_ms logging to those scripts; re-measure over all samples (cheap)
Inconsistent timing boundaries	Cross-day comparison isn't apples-to-apples even after call-count normalizing	Standardize the timing boundary (generate-only, with cuda.synchronize) across all metrics
Warm-up / async-timing skew	Small calibration means overstated Day 6/7 cost	Discard warm-up iterations; explicit torch.cuda.synchronize() around timed regions
Linear extrapolation, no batching	Under-uses hardware; not a real throughput	Measure batched throughput on target GPU; treat linear as an upper bound
RTX-3070-specific + eager-attention path	Won't transfer to cluster; attention overhead is hardware-dependent (SDPA vs eager)	Re-benchmark on deployment hardware; note the attention-routing caveat from the feedback doc
Attribution is the costliest stage	output_attentions=True forces the slow path on every attributed sample	Subsample attribution, or use a faster attention-capture route; it's the single biggest optimization lever
Budget is for the uncorrected pipeline	The real study adds reruns + possibly gradient-based attribution	Cost decision-1 options explicitly — gradient/LRP methods are materially more expensive than forward passes
Streaming I/O could dominate	"Compute is trivial" hides an I/O bottleneck if streaming	Local caching (already done — loader now reads local parquet), keeping the run GPU-bound
The one reassurance that separates this chapter
Unlike Chapters 4–7, the conclusion survives all its caveats: the chapter's own argument is that the headline "would have to be wrong by close to an order of magnitude before compute became a concern," and the sanity check found no such gap. So even granting every measurement weakness above, compute affordability doesn't flip. The fixes here buy precision and hardware-portability, not a different answer — which is why the meeting package (decision 4) correctly frames the real scaling question as "full split vs. fixing the struck_by_risk imbalance," explicitly noting compute is not the deciding factor.

Through-line note: this chapter is where the Chapter 6 decision (attribution method) shows a hidden cost dimension — the choice isn't only about validity (is cross-attention acceptable evidence?) but about efficiency (a gradient-based alternative could dominate the entire compute budget). Those two should be decided together.

Chapter 10
Analysis for Chapter 9 (Robustness) — the richest chapter, and scientifically the strongest. Its centerpiece (Finding 2) is the pilot's best result, so the issues here are mostly about coverage, contamination, and one cross-chapter validity risk rather than broken logic.

What it does
Three test families with deterministic decoding (to isolate perturbation from Ch7's sampling noise — a good choice): Level 1 (blur, low_light, occlude, contrast_shift × 163 = 652 reruns), Level 2 (reworded prompt, 138 samples, rule_4 honestly excluded), and a 20-sample synthetic hard-hat-patch stretch test. Four diagnostic columns per rerun: answer_changed, confidence_drop, object_box_iou, worker_lost.

Its methodological contributions are genuinely strong and should be kept: the two-number (answer-level vs explanation-level) split, the worker_lost diagnostic, the denominator correction, and holding decoding deterministic.

Problems this chapter surfaced (and carries)
1. The fallback artifact, third named appearance. Occlude's 28.2% answer-change includes 9/46 flips with worker_lost=True — the same degenerate scene-level rerouting from Ch5, now triggered by a perturbation. Image 0000341: occlusion that doesn't even fully cover the boxes still kills both detections → fallback fires → flip. So raw answer_changed over-counts genuine robustness failures; the chapter flags this but doesn't subtract it from the headline.

2. Centerpiece — stable answer, drifted explanation. 28.3% of stable-answer rows have object_box_iou < 0.3 (blur 58.8%, reworded 56.7%); image 0000034 has object_box_iou = 0.005 with the answer unchanged. This is the pilot's strongest argument that answer-level metrics are necessary but not sufficient.

3. Confidence proxy is uncalibrated — it rose under the two most disruptive perturbations. Confidence tracks decoder commitment, not correctness.

4. rule_2's two "phrasings" aren't synonyms — "harness" vs "lanyard" are different physical objects, so its 50% rewording-change rate measures a prompt-design flaw, not model instability.

Weaknesses in the setup (what could hinder scaling)
5. Every perturbation is a single, arbitrary magnitude. blur ksize=8.0, gamma 2.5, occlude frac=0.2 centered, contrast 0.3. The whole robustness table is a function of these fixed points — there's no dose-response curve, so you can't tell whether 27% blur-change is fragile or resilient.

6. Occlusion conflates two things. The centered fixed-fraction black square (a) reintroduces the OOD black-patch artifact from Ch4/5, and (b) Finding 1 shows it disrupts whole-scene grounding, not just the covered pixels. So "occlude" isn't cleanly measuring "robustness to a covered region."

7. object_box_iou = NaN excludes the worst cases. Finding 2's denominator is only rows where both sides detected an object box. Cases where the box vanishes entirely — arguably the worst explanation instability — drop out of the 28.3% (counted separately as worker_lost, but not in the drift number). The two need joint reporting, or 28.3% reads as the whole story when it's a floor.

8. ⚠️ Cross-chapter validity risk: Finding 2 may be partly the Chapter 7 size confound. object_box_iou is the same size-sensitive metric — a small box (hard hat) drops below IoU 0.3 from a few pixels of jitter, exactly the mechanism Ch7 identified. Image 0000034 (iou=0.005, box relocated across the frame) is a genuine relocation, but the aggregate 28.3% at threshold 0.3 could be inflated by small-box IoU sensitivity rather than true evidence drift. This is the one place a headline finding might be partly a mechanical artifact, and it should be checked before the paper leans on the number.

9. worker_lost catches only one failure mode (worker present→absent); "worker still found but relocated" (image 34) slips through — which is precisely why object_box_iou was needed.

10. Stretch test is crude and underpowered — non-photorealistic patch (yellow ellipse on beige), geometric head-box guess with no pose estimation (image 233 landed on the torso), first-worker only, n=4 meaningful. All acknowledged; it's a plausibility probe, not a result (decision 5).

Hindrances to scaling, and how to solve each
Hindrance	Why it bites at scale	Fix
Single arbitrary perturbation magnitudes	Can't tell robust from fragile; one point isn't a curve	Severity sweeps — report answer/explanation change vs perturbation strength (dose-response)
Fallback-rerouting contaminates answer_changed	Raw rate over-counts genuine failures (Ch5/Ch10 mechanism)	Segment out worker_lost flips; report corrected perturbation sensitivity — same instrumentation as Ch5/Ch10
⚠️ Finding 2 may include Ch7 size artifact	The centerpiece result could be partly IoU-size sensitivity	Recompute the 28.3% controlling for object size (does drift survive within size bands?); add a size-invariant drift measure
object_box_iou NaN drops vanished boxes	Worst instability invisible in the drift number	Report drift and box-disappearance jointly
Occlusion conflates coverage with global disruption	"occlude" isn't measuring one thing	Split into targeted occlusion (cover the actual object box) vs random-location; consider blur-occlusion to cut the black-patch OOD artifact
Rewording set thin; rule_2 not synonymous	Rewording robustness confounded by concept mismatch	Build a validated paraphrase bank; treat rule_2's harness/lanyard as two concepts (decision 6)
Explanation drift = box IoU only	One scalar under-describes drift	Add attention-heatmap drift (Ch6) as a second explanation-level signal
Stretch test crude & underpowered	Can't claim a spoofing vulnerability from n=4	Real hard-hat patches, pose-based placement, powered n (decision 5)
First-object-box comparison	Reorder artifact if multiple boxes	Match boxes (IoU/Hungarian) before computing drift (Ch7 showed reorder is small but nonzero)
Compute is not the blocker (Ch8): Level 1 is 652 reruns; even a full severity sweep across 5 magnitudes stays well within budget.

The through-line, now consolidated
This chapter closes the loop on the pilot's two recurring mechanisms. The fallback-rerouting artifact has now appeared under all three triggers it can have — deliberate masking (Ch5), perturbation (Ch9 Finding 1), and completeness reruns (Ch10). And the size confound (Ch6, Ch7) reaches into this chapter's headline finding via object_box_iou. That second link is the most important thing to verify: if you fix the Ch4 ranking and add a size-invariant drift measure, you simultaneously de-risk Chapters 6, 7, and 9's Finding 2.

Want me to recompute Finding 2's 28.3% controlling for object size — bucketing the stable-answer rows by object-box area and checking whether the sub-0.3-IoU drift rate survives within each size band — so we know how much of the pilot's centerpiece result is genuine explanation drift versus the Chapter 7 small-box IoU artifact? It reuses robustness.csv and baseline_predictions.csv, no model reruns needed.

Chapter 11: Aggregating Six Metrics into a Single Cross-Class Picture
This chapter shows immense analytical discipline. Instead of just throwing all six metrics into a chart and claiming victory, you actively guarded against overstating what the numbers mean
.
The "False Confirmation" Catch (Finding 1): You correctly diagnosed that Descriptive Accuracy and Bounded Completeness sharing the same class shape is not independent confirmation, because your completeness metric is computationally built from the descriptive accuracy answer-change columns
. This level of transparency will easily pass peer review.
The Sparsity vs. Completeness Dichotomy (Finding 2): Your observation on the struck_by_risk class is brilliant. It scored the highest on descriptive accuracy (61.5%) but the lowest on visual sparsity (0.067)
. You rightly attribute this to the fact that rule 4 targets an excavator—a massive structural object that geometrically forces attention to spread out
. This proves to Abdallah that the metrics measure genuinely different properties, not just one redundant "explanation quality" score
.
Unmasking the fall_hazard Robustness (Finding 3): You noticed that fall_hazard has the worst robustness (75.0%) but the second-best descriptive accuracy
. You expertly traced this back to the fact that the class pools two sub-rules with entirely different failure modes: rule 2's worker-lost artifact and rule 3's grid-fallback error
.
Chapter 12: Synthesizing Findings into a Pilot Report
Your editorial policy in this chapter is exactly what top-tier journals require.
Sourcing Discipline: By strictly demanding that every single number trace back to a specific CSV rather than relying on memory
, you ensure the integrity of the data.
Cross-Cutting Synthesis: This is where the magic of your report happens. By reading the 11 days side-by-side, you successfully unified the scattered anomalies into two grand, overarching mechanisms: the fallback-rerouting mechanism (spanning Chapters 5, 9, and 10) and the area-based region-ranking confound (spanning Chapters 4, 6, 7, and 10)
. This prevents the paper from looking like a list of random bugs and instead presents a cohesive story of architectural interactions
.
Chapter 13: From-Scratch Reproducibility
Running a clean-room reproducibility check on a fresh 20-sample subset
 is a masterclass in software engineering for AI research.
Catching the Hardcoded Bug: Because you purposefully chose n=20 (5 samples per class) instead of your original 163, you successfully exposed a hidden bug in 11_make_figures.py where the chart labels were hardcoded to literal strings (like n=50 and n=13)
. If you had only tested on the 163-sample set, this would have slipped into your final paper, leading to mismatched labels
.
The Network Streaming Blip: During this run, your script threw an IncompleteRead/ProtocolError from Hugging Face, which was handled by auto-retry
. This perfectly validates the warning I gave you during our Chapter 1 review: when you scale up to the high-performance cluster for 3,000+ images, you must cache the dataset locally. If you rely on streaming across thousands of cluster GPUs, this exact network blip will exhaust your retry budget and crash your cluster runs
.
Chapter 14: Packaging the Pilot for the Meeting
You have packaged this perfectly for Professor Abdallah. Your headline answer—that his framework does transfer to VLMs, but strictly required your grounding-proxy reframing and person-relative redesign to function
—positions you as an equal collaborator who adapted his work, rather than just a student running his code.
The "Six Decisions" Strategy: Framing your future work as six explicit decisions for him to weigh in on is the best way to run this meeting
. I recommend you push hardest on Decisions 1 and 3:
Decision 3 (The Area-Ranking Heuristic): You must get alignment to fix this
. As you've proven across four chapters, ranking by bounding-box area ruins the evaluation of small PPE items (like hardhats), and implementing a semantic, rule-aware ranking is the highest-leverage fix you can make before the cluster run
.
Decision 1 (The Attribution Method): You need his blessing on your use of decoder-encoder cross-attention
. His own recent 2024 and 2025 papers rely entirely on strict white-box methods like Layer-wise Relevance Propagation (LRP), Integrated Gradients (IG), and DeepLIFT
. Because you proved that Florence-2's fused image-text encoder breaks traditional ViT rollout
, you need to ensure he accepts cross-attention as mathematically valid for his framework, or ask if he knows a way to force LRP/DeepLIFT through a fused multimodal Transformer
.