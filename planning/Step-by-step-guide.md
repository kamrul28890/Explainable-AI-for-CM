Here is a step-by-step guide on how to prepare and approach him for this collaboration:
Step 1: Study Professor Abdallah’s Core Methodologies You do not need to become a cybersecurity expert, but you must understand how he evaluates AI models. Focus on reading his two recent papers provided in your sources:
The E-XAI Framework Paper: Read how he evaluates black-box XAI methods (SHAP and LIME) using his six core metrics: Descriptive Accuracy, Sparsity, Stability, Efficiency, Robustness, and Completeness
. Understand how he defines these metrics, because your pitch will be to apply these exact metrics to your construction VLMs.
The White-Box XAI Paper: Read his work comparing White-box methods (Layer-wise Relevance Propagation (LRP), Integrated Gradients (IG), and DeepLift)
. Note his conclusion that white-box methods often score higher in robustness and completeness than black-box methods
. This is highly relevant since your research notes that VLMs in construction suffer from environmental noise and reliability issues
.
Step 2: Map Your Datasets to His Frameworks Professor Abdallah tests his XAI frameworks on network intrusion datasets like NSL-KDD, CICIDS-2017, and RoEduNet-SIMARGL2021
. To collaborate, you need to bring the construction equivalent of these datasets.
Look at the datasets mentioned in your own literature review (e.g., SODA, ACID, VSD, or custom photologs)
.
Prepare a list of 2 or 3 specific multimodal construction datasets where VLMs are currently used for Visual Question Answering (VQA) or safety compliance
. You will tell him: "You test XAI on network traffic logs; I want us to test it on spatiotemporal construction video logs."
Step 3: Draft a 1-Page Research Pitch Professors are busy, so before you meet him, synthesize the research proposals we discussed into a concise 1-pager. Highlight the synergy between your fields:
The Problem: Construction VLMs suffer from the "semantic gap" and hallucinate under dynamic site noise
.
The Solution: Applying his exact E-XAI evaluation framework
 or White-box XAI methods
 to quantify which visual pixels or text prompts the VLMs are actually relying on when they make safety predictions.
Your Contribution: Clearly state that you will handle the VLM deployment, dataset preparation, and construction domain knowledge, asking him to guide the XAI integration and robustness testing.
Step 4: Reach Out and Schedule a Meeting Once you have your 1-pager and a basic understanding of his 6 metrics, reach out to him.
According to his Purdue Polytechnic profile, his email is abdalla0@purdue.edu and his office is located at ET 201J in Indianapolis
.
Send a short, professional email stating that you are a Construction Management student focusing on Vision-Language Models, that you have read his recent papers on evaluating XAI robustness
, and that you have a concrete proposal to apply his E-XAI framework to a novel domain (construction safety). Ask for a 15-minute meeting to discuss it.
Step 5: Propose a Small Pilot Experiment During your meeting, offer to run a quick proof-of-concept. For example, tell him you will take a small batch of construction images (e.g., workers missing hardhats), run them through a VLM to get hazard predictions, and attempt to apply SHAP or LIME to those predictions
. Ask if he would be willing to review your initial XAI output. Starting with a small, actionable pilot shows initiative and makes it easy for him to say "yes" to guiding you.