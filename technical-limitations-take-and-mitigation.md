# Technical Limitations: My Take and How to Address Them

## Executive Take

The limitations listed in `technical-limitations.md` are real. They do not weaken the research idea, but they do require a tighter technical framing.

The main issue is that the current proposal sometimes sounds as if Professor Abdallah's exact tabular-DNN XAI pipeline can be directly transferred to Florence-2 or another VLM. That is not technically true. His framework can be transferred at the evaluation level, but the implementation details must change because VLMs are Transformer-based, multimodal, memory-heavy, and often generative.

The best way to address this is to separate the project into two layers:

1. **Framework transfer:** Keep Abdallah's six evaluation metrics.
2. **Implementation adaptation:** Use VLM-appropriate attribution methods and bounded pilot tests.

This distinction is important. We are not claiming that his exact code or exact DNN workflow will run unchanged on Florence-2. We are claiming that his evaluation logic can be adapted to construction VLMs in a technically careful way.

## 1. Architecture Compatibility Gap

### My Take

This is the most important limitation. LRP, IG, and DeepLIFT were easier to apply in Abdallah's work because the models were standard neural networks over tabular data. Florence-2 and other VLMs are much more complex. They include visual encoders, attention layers, token embeddings, cross-modal alignment, and text generation components.

It would be risky to promise that legacy white-box XAI tools such as iNNvestigate will work directly on Florence-2. They probably will not work without custom engineering.

### How to Address It

The proposal should not say that we will directly apply traditional DNN white-box packages to Florence-2. It should say:

> We will adapt the six-metric XAI evaluation framework to VLMs using Transformer-compatible attribution methods. The pilot will begin with attention rollout, Grad-CAM-style visual attribution, Integrated Gradients where supported, and occlusion-based region testing. LRP and DeepLIFT will be explored later only where the model architecture and implementation support them.

### Practical Plan

For the pilot:

- Use **Florence-2** for grounding and detection-style outputs.
- Use **occlusion and region masking** as the most reliable first explanation test.
- Use **attention rollout or gradient-based attribution** only if the implementation gives stable access to intermediate activations and gradients.
- Treat LRP and DeepLIFT as second-stage methods, not pilot guarantees.

### Proposal Change Needed

Replace strong claims like:

> We will deploy IG, LRP, and DeepLIFT directly onto Florence-2.

With:

> We will begin with VLM-compatible attribution methods, including attention-based, gradient-based, and perturbation-based approaches. Architecture-specific white-box methods such as LRP and DeepLIFT will be added where technically feasible.

## 2. Robustness Test Contradiction

### My Take

This limitation is correct. Abdallah's original robustness test depends on training biased and adversarial models. A 2 to 4 week pilot using a pretrained VLM cannot honestly reproduce that exact setup.

However, this is not a fatal problem. The robustness concept can still be tested. We just need to define a VLM-appropriate robustness test.

### How to Address It

For the pilot, robustness should be tested through input-level attacks rather than model-training attacks.

Use:

- visual occlusion,
- blur,
- lighting changes,
- contrast changes,
- synthetic PPE patches,
- fake safety markings,
- misleading prompts,
- prompt wording changes,
- irrelevant text added to the prompt.

This tests whether the explanation can be pulled away from the true hazard cue.

### Practical Plan

Create three robustness levels:

| Level | Test Type | Example |
|---|---|---|
| Level 1 | Natural visual noise | blur, low light, occlusion |
| Level 2 | Semantic prompt noise | misleading wording or added irrelevant details |
| Level 3 | Adversarial visual cue | fake PPE sticker, false harness-like patch, misleading sign |

The pilot should only promise Level 1 and a small part of Level 2. Level 3 can be proposed as the full-study extension.

### Proposal Change Needed

Say:

> Because the pilot uses pretrained VLMs without retraining, robustness will be tested through controlled visual and prompt perturbations rather than through retraining biased and adversarial models.

This shows the professor that we understand his original robustness design but are adapting it responsibly.

## 3. Memory Overload During Completeness Testing

### My Take

This is a serious computational risk. Completeness testing can become expensive very quickly because it may require repeated attribution, perturbation, and re-inference. Doing this over full-resolution images and VLMs can exceed normal GPU memory limits.

The proposal should not promise full completeness testing over every pixel, every patch, or every possible perturbation.

### How to Address It

Use a bounded completeness test.

For the pilot:

- test only top-1 and top-2 attributed regions,
- use bounding boxes or superpixels instead of raw pixels,
- use a small sample set,
- cache model outputs,
- run attribution in batches,
- use fixed image resolution,
- avoid repeated full-gradient extraction when an occlusion test is enough.

### Practical Plan

Define completeness as:

> A sample passes the bounded completeness check if the model provides a usable explanation and if perturbing the top-ranked region or top two regions causes a measurable change in the model's answer, confidence proxy, or grounding output.

This is not full theoretical completeness. It is a practical pilot version.

### Proposal Change Needed

Say:

> The pilot will use a bounded completeness test. Instead of perturbing all possible image features, we will test whether the top one or two attributed visual regions affect the VLM's output when masked or blurred.

This keeps the metric meaningful while avoiding an impossible compute promise.

## 4. Multimodal Feature Tuple Math

### My Take

This is also correct. Visual attributions and text-token attributions do not naturally live in the same mathematical space. Pixel gradients, image-patch scores, bounding-box scores, and token-embedding gradients cannot be merged casually into one score without a clear normalization strategy.

The current idea of "visual region + semantic query" is still useful, but the metrics should be computed separately first.

### How to Address It

Separate the explanation streams:

1. **Visual attribution stream**
   - image patches,
   - bounding boxes,
   - object regions,
   - motion regions.

2. **Text attribution stream**
   - prompt tokens,
   - safety-rule terms,
   - question phrases.

Then report:

- visual sparsity,
- text sparsity,
- visual stability,
- text stability,
- cross-modal agreement.

Cross-modal agreement is the bridge metric. It asks whether the model's visual evidence and prompt evidence refer to the same safety concept.

Example:

| Safety Question | Expected Visual Evidence | Expected Text Evidence | Good Explanation |
|---|---|---|---|
| Is the worker wearing a hardhat? | worker head region | hardhat token | Both visual and text attribution focus on hardhat compliance |
| Is the worker tied off? | harness/lanyard/anchor region | tied off, harness, anchorage tokens | Both streams focus on fall-protection evidence |

### Proposal Change Needed

Say:

> Visual and textual attributions will be normalized and evaluated separately. We will report visual sparsity and text sparsity as separate metrics, then examine cross-modal agreement between highlighted regions and safety-rule tokens.

This is much stronger than forcing one combined score.

## Recommended Revised Technical Framing

The proposal should use this framing:

> This project adapts Abdallah's six-metric XAI evaluation framework to construction VLMs. The metrics remain the same, but the attribution implementation is modified for Transformer-based multimodal models. The pilot will use bounded, VLM-compatible tests based on visual grounding, region masking, attention or gradient attribution where feasible, prompt perturbation, and separate visual/text explanation scoring.

This sentence solves most of the technical risk.

## Revised Pilot Design

The pilot should be scoped like this:

1. Use 100 to 300 ConstructionSite 10k samples.
2. Use Florence-2 as the grounded VLM.
3. Use safety-rule VQA prompts.
4. Generate model answers and grounding outputs.
5. Use region masking as the primary explanation test.
6. Use attention or gradient attribution only where technically stable.
7. Compute:
   - descriptive accuracy,
   - visual sparsity,
   - visual stability,
   - runtime efficiency,
   - small robustness test,
   - bounded completeness test.
8. Leave full LRP, DeepLIFT, full adversarial training, and large-scale completeness for the full study.

## What to Tell the Professor

The technically mature pitch is:

> We do not assume that the existing tabular-DNN XAI code will transfer directly to VLMs. The contribution is to transfer the evaluation framework and adapt the implementation to Transformer-based construction VLMs. The pilot will start with bounded visual grounding and region-perturbation tests, then expand toward deeper white-box attribution where the model architecture permits.

That is the right balance. It shows ambition, but it does not overpromise.

## Final Recommendation

Keep the proposal, but revise the technical claims in four places:

1. Replace direct LRP/DeepLIFT promises with Transformer-compatible attribution language.
2. Redefine robustness as visual and prompt perturbation for the pilot.
3. Make completeness bounded to top-1 and top-2 regions.
4. Separate visual and text attribution metrics before discussing cross-modal agreement.

With these changes, the proposal becomes more credible. It also gives Professor Abdallah a clear role: helping translate his six-metric evaluation logic into a technically valid VLM setting.
