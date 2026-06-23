# Task List for a Full Study — What We'd Want Your Guidance On

Each item below is something the pilot deliberately scoped down or left undone, not something it tried and failed at. Framed as decisions, not just to-dos, because each one needs Professor Abdallah's judgment before it's worth committing engineering time to.

## 1. Attribution method

- [ ] **Decide**: is decoder→encoder cross-attention an acceptable stand-in for LRP/DeepLIFT in the framework's terms, or does a transformer-native attribution method need to be added?
- If yes to a transformer-native method: scope which one (e.g. integrated gradients on the vision encoder, a transformer-specific LRP variant) and whether it needs to run per-rule or once per image.
- Pilot evidence: Day 6 (`day6_findings.md`) — classic attention rollout's single-modality assumption verified not to hold for Florence-2's fused encoder; cross-attention used instead.

## 2. Statistical rigor at small n

- [ ] **Decide**: what's the minimum acceptable n per class before a metric is reported as a stable estimate, and what test/interval should accompany each headline number going forward.
- Directly blocks trusting any `struck_by_risk` number as currently reported (n=13).
- Pilot evidence: `pilot_report_draft.md` §7, first bullet.

## 3. Region-ranking heuristic

- [ ] **Decide**: should regions be ranked by "matches the rule's queried object class" before falling back to raw area, to remove the worker-vs-PPE-object confound?
- This single fix plausibly improves three metrics' weakest results (completeness, stability, sparsity) simultaneously for PPE classes — worth prioritizing before scaling up sample size, since scaling up would just give a larger, equally-biased sample.
- Pilot evidence: `pilot_report_draft.md` §6.2 — 96% of `ppe_violation` samples rank the worker's box ahead of the PPE item; 0.499 vs. 0.922 stability IoU (PPE vs. excavator).

## 4. Full-dataset scaling vs. class-imbalance fix

- [ ] **Decide**: is scaling to the full 3,004-image test split the right next step, or is fixing the `struck_by_risk` shortage (only 13 examples exist in the entire split) a higher-value use of the same effort?
- Compute is confirmed not the constraint either way (~0.5 GPU-hours per 1,000 samples, Day 8).
- Pilot evidence: `data/pilot_samples.csv` value counts; `pilot_report_draft.md` §6.5, §8.4.

## 5. Real Level-3 robustness study

- [ ] **Decide**: scope and budget for a properly powered adversarial-patch study (optimized patch placement, larger n, possibly multiple patch types).
- The pilot's 20-image stretch task only had 4 cases where "fooling" was even a meaningful test, and the result didn't show a clean placement→flip correlation (a well-placed patch failed, a badly-placed one succeeded) — a real signal, not a result.
- Pilot evidence: `day9_findings.md`, "Stretch" section.

## 6. rule_2 prompt-phrasing ambiguity

- [ ] **Decide**: should rule_2's two phrasings ("safety harness" vs. "fall-protection lanyard") be treated as two distinct concepts rather than wording variants, before rule_2's robustness numbers are trusted in a larger study?
- Pilot evidence: `day9_findings.md`, Finding 3 — 50% Level-2 answer-change rate for rule_2 vs. 10.2% for rule_3, traced to the two phrasings plausibly grounding different physical objects, not just different wording of the same one.

## Out of scope for this pilot, unchanged from the original plan

Full adversarial model retraining, full CMA temporal/video integration, SODA benchmark integration — none of these were attempted or partially attempted; they remain fully open for a future study, not summarized further here.
