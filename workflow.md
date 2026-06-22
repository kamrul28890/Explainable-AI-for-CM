Step 1: Environment Setup & Data Subsampling
You need to isolate the exact dataset you promised in the proposal.
Action: Pull 100–300 balanced samples from the ConstructionSite 10k dataset
. Ensure you have a roughly even split of your target classes: compliant scenes, PPE violations, fall hazards, and struck-by/proximity risks
.
Tooling: Set up your Python environment with Hugging Face transformers, torch, and load the Florence-2-base-ft model
.
The Feature Tuples: For each of the 300 images, define the exact semantic VQA prompt you will ask the model (e.g., "Is the worker wearing a hardhat?").
Step 2: Baseline Inference (The "Black-Box" Run)
Before you can explain the model, you need to record what it naturally predicts.
Action: Feed the 300 image-prompt pairs into Florence-2.
Data to Capture: For every run, log the predicted answer (e.g., "Yes" or "No"), the model's confidence score, and the generated visual grounding coordinates (the bounding boxes Florence-2 natively outputs)
.
Step 3: Transformer-Specific Gradient Extraction (The XAI Integration)
This is where we address the architectural weakness we discussed earlier. We cannot use Professor Abdallah's legacy iNNvestigate library because it frequently crashes under heavy memory loads and struggles with models lacking simple softmax layers
.
Action: Instead of traditional Layer-wise Relevance Propagation (LRP), implement Attention Rollout or Gradient x Attention mapping
.
Tooling: Use a modern attribution library like PyTorch's Captum. You will extract the gradient scores from Florence-2's visual encoder and its text decoder independently to see exactly which image patches and which prompt tokens fired the hardest during the prediction.
Step 4: Computing the First Two E-XAI Metrics
Once you have the gradient scores, you need to calculate the actual metrics to present to Professor Abdallah.
Compute Visual Sparsity: Look at the visual gradients. Does the attribution mass concentrate tightly on the worker/hazard, or is it scattered across the background clutter?
. Calculate this independently from the text prompt sparsity.
Compute Descriptive Accuracy: Take the top k image patches highlighted by the XAI method and computationally mask them (turn those pixels black or blur them)
. Feed the masked image back into Florence-2. If the model's confidence drops significantly or the answer changes, record this as a successful Descriptive Accuracy score
.
Step 5: The Mini-Robustness Test (Adversarial Patching)
To prove you understand his cybersecurity frameworks, you need to run a quick deception test without retraining the model weights.
Action: Take 20 of your "PPE Violation" images. Computationally paste a synthetic, highly realistic "hardhat" or "safety harness" patch onto the worker.
The Test: See if the fake patch tricks Florence-2 into predicting "Safe." If it does, run the XAI method to see if the explanation successfully highlights the fake patch as the reason for the deception
.