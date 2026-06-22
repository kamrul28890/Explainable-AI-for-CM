# Research Proposal: Evaluating Explainability in Vision-Language Models for Dynamic Construction Safety Monitoring

## 1. Introduction and Motivation

Construction sites are complex and change quickly. Workers, equipment, materials, temporary structures, and environmental conditions interact in ways that make safety monitoring difficult. Manual inspections are important, but they are also time-consuming, subjective, and limited by what an inspector can observe at a given moment.

Traditional computer vision systems can detect objects such as hardhats, workers, excavators, ladders, or scaffolds. However, object detection alone does not explain whether a scene is safe. A model may detect a worker and a harness, but still fail to answer the safety question that matters: Is the worker tied off correctly while working at height?

Vision-Language Models (VLMs) offer a useful next step because they combine visual input with natural-language questions. They can support tasks such as:

- answering safety questions about a site image,
- generating captions for observed hazards,
- grounding a safety answer to specific image regions,
- checking whether a scene appears compliant with a safety rule.

The problem is that VLMs can still make unreliable safety decisions. Their outputs can change when prompts are reworded, when lighting changes, when objects are partly hidden, or when the site is visually cluttered. In a safety setting, it is not enough for a model to say that a scene is safe or unsafe. A safety manager also needs to know what visual evidence the model used.

This project proposes an explainability evaluation framework for VLMs in construction safety. The framework adapts Professor Mustafa Abdallah's six-metric XAI evaluation approach from network intrusion detection to multimodal construction data. The goal is to test whether VLM explanations are accurate, focused, repeatable, efficient, resistant to noise, and valid across difficult safety cases.

Rather than claiming that every VLM can be fully opened and explained, this project will begin with open-source VLMs where internal visual representations can be inspected. The pilot will use gradient-based and attention-based attribution methods first, such as Integrated Gradients, Grad-CAM-style attribution, and attention rollout. More architecture-specific white-box methods, such as Layer-wise Relevance Propagation and DeepLIFT, will be added where the model structure makes them technically feasible.

## 2. Data Foundation

The proposed study will use a focused dataset stack that supports construction safety monitoring across spatial and temporal conditions.

### 2.1 Dataset Stack

| Dataset | Role in the Study | Why It Is Useful |
|---|---|---|
| ConstructionSite 10k | Primary safety VLM dataset | Provides construction-site images with safety-rule VQA, captions, grounding, bounding boxes, and image attributes. This is the strongest dataset for testing whether VLM answers are tied to the right visual evidence. |
| SODA | Spatial clutter and object benchmark | Provides a large set of construction-site images with object annotations across varied layouts, weather, viewpoints, and site conditions. This supports object-level attribution and robustness testing. |
| CMA, Construction Meta Action | Temporal safety behavior benchmark | Provides video clips of construction worker actions. This supports testing whether explanations remain useful when safety depends on motion or repeated behavior over time. |

This stack keeps the project focused on dynamic safety monitoring. It avoids broadening the proposal into daily reporting or progress monitoring before the safety explainability question is well defined.

### 2.2 Construction Anomaly Classes

Professor Abdallah's work evaluates model explanations for normal network traffic and attack classes. In this project, the construction equivalent is safe site behavior versus safety anomalies.

| Class | Construction Meaning | Example |
|---|---|---|
| Compliant or safe state | No visible violation in the safety rule being checked | Worker has required PPE, equipment is separated from workers, access path is clear |
| PPE violation | Required protective equipment is missing or used incorrectly | No hardhat, no vest, harness visible but not tied off |
| Fall hazard | Worker is exposed to fall risk | Worker near unprotected edge, unsafe ladder use, missing guardrail |
| Struck-by or caught-between risk | Unsafe worker-equipment or worker-material interaction | Worker inside equipment swing radius, worker close to moving machine |
| Site obstruction or congestion | Unsafe site layout or blocked movement | Material pile blocks access, equipment crowds work area |
| Repeated unsafe action | Unsafe behavior persists over time | Repeated unsafe climbing, repeated close proximity to equipment, repeated missing PPE across frames |
| Subtle or occluded violation | Hazard is present but visually difficult | Worker partly hidden by dust, distance, glare, or overlapping equipment |

For the first pilot, the label set should stay small:

- compliant or safe,
- PPE violation,
- fall hazard,
- struck-by or caught-between risk.

This limited scope is easier to test and easier to explain to the professor.

## 3. Core Methodology

The key translation is from tabular network features to multimodal construction features.

In Professor Abdallah's network security work, explanation methods rank features such as flow duration, destination port, packet length, or protocol type. In construction VLMs, the feature units are different:

- visual regions, such as bounding boxes, image patches, object masks, or worker-equipment interaction zones,
- temporal features, such as object motion, action clips, or repeated unsafe states across frames,
- text features, such as safety-rule prompts, question tokens, and hazard-related words.

This project will treat the explanation unit as a multimodal feature pair. For example:

- worker head region plus hardhat rule,
- harness region plus tie-off question,
- equipment swing area plus worker proximity rule,
- ladder region plus fall-protection prompt.

This gives the XAI framework a clear target. The explanation should show which visual regions and which rule concepts influenced the VLM's safety answer.

## 4. Six-Metric XAI Evaluation Plan

The study will evaluate VLM explanations using six metrics adapted from Professor Abdallah's framework.

### 4.1 Descriptive Accuracy

This metric tests whether the explanation identifies features that actually matter to the model.

Process:

- Rank the most important visual regions and text tokens.
- Mask or blur the top-ranked visual regions.
- Remove or reword the top-ranked safety-rule tokens.
- Measure whether the VLM answer changes or confidence drops.

If removing the highlighted features changes the answer, the explanation is more likely to be faithful to the model's behavior.

### 4.2 Sparsity

This metric tests whether the explanation is focused.

A useful safety explanation should concentrate on the hazard region, not the whole image. For example, if the question is about hardhat compliance, the explanation should focus on the worker's head region, not on background scaffolding or unrelated equipment.

### 4.3 Stability

This metric tests whether the explanation is repeatable.

The same image and the same safety question will be submitted multiple times. The top visual regions and text tokens should remain similar across runs. Stability can be measured with overlap scores, Intersection over Union for highlighted regions, or rank correlation for feature lists.

### 4.4 Efficiency

This metric tests whether the explanation can be generated within a practical time budget.

The study will measure:

- explanation runtime,
- memory use,
- number of forward and backward passes,
- throughput across a batch of images.

This matters because a future safety tool may need to run on site laptops, tablets, or edge devices.

### 4.5 Robustness

This metric tests whether explanations remain useful when the input is noisy or misleading.

The study will add controlled perturbations such as:

- blur,
- low light,
- overexposure,
- partial occlusion,
- dust-like noise,
- viewpoint shift,
- misleading visual cues such as false or unclear PPE evidence.

The goal is to test whether the model explanation still points to the real safety cue, or whether the explanation is pulled toward irrelevant noise.

### 4.6 Completeness

This metric tests whether the explanation remains valid across all relevant sample types, including difficult cases.

The study will examine whether each sample receives a usable explanation and whether perturbing the top-ranked features changes the model's safety answer. If the model says that a region is important, but removing that region does not affect the answer, the explanation may be incomplete or unfaithful.

Completeness will be tested especially on:

- crowded scenes,
- overlapping worker-equipment interactions,
- partial occlusions,
- subtle PPE misuse,
- multiple hazards in one image,
- video clips where the hazard depends on motion.

## 5. Proposed 2 to 4 Week Pilot Experiment

The pilot should be small and practical. It should show that the framework can run on real construction data without requiring full model training.

### 5.1 Pilot Dataset

Use 100 to 300 balanced samples from ConstructionSite 10k.

Suggested sample groups:

- compliant scenes,
- PPE violations,
- fall hazards,
- struck-by or proximity risks.

### 5.2 Pilot Model

Use Florence-2 as the first pilot model.

Reason:

- it is lightweight enough for a short pilot,
- it supports grounding and detection-style outputs,
- it is useful for connecting safety answers to image regions,
- it gives a practical starting point for visual attribution.

Qwen2-VL can be kept as a second-stage comparison model because it is stronger for general VQA and broader visual reasoning. However, Florence-2 is the better first model for a fast feasibility test.

### 5.3 Pilot Steps

1. Select 100 to 300 samples from ConstructionSite 10k.
2. Define a small set of safety-rule questions.
3. Run Florence-2 on each image and prompt.
4. Store the model answer, caption, grounding output, and confidence proxy if available.
5. Generate visual attributions using gradient-based or attention-based methods.
6. Mask top-ranked visual regions and re-run the model.
7. Compute descriptive accuracy, sparsity, stability, and efficiency.
8. Add a small robustness test using blur, lighting change, and occlusion.
9. Compare highlighted regions against ground-truth boxes or expected safety regions.

## 6. Expected Contributions

This project can make three clear contributions.

1. A construction-specific XAI evaluation framework for VLM safety decisions.
2. A dataset mapping that connects construction safety anomalies to Professor Abdallah's normal-versus-attack evaluation logic.
3. A pilot benchmark showing whether VLM explanations remain accurate, focused, stable, efficient, and robust under construction-site conditions.

The long-term deliverable is a journal paper that evaluates whether open VLMs can provide trustworthy explanations for safety-critical construction monitoring. A suitable target venue would be Automation in Construction, with possible alternatives in IEEE Access or related AI-for-construction venues.

## 7. Collaboration Rationale

This project fits both research areas.

The construction-management side contributes:

- construction safety context,
- domain-specific hazard classes,
- construction datasets,
- VLM use cases,
- interpretation of site conditions.

Professor Abdallah's side contributes:

- XAI evaluation methodology,
- experience with black-box and white-box explanation methods,
- robustness and completeness testing,
- statistical evaluation of explanation quality.

The proposed collaboration is therefore not just applying a VLM to construction images. It is testing whether the explanations behind VLM safety decisions can be trusted.
