# §5.5 annotation audit -- labelling page

Published privately at https://claude.ai/artifact/By1Bw7mXqj94pQU1sJBbMg
(share it with the auditor as Contributor or above so answers save).

- `scripts/12_audit_sample.py --export DIR` draws the 100 images
  (`results/audit_manifest.csv`; per quality tier 25 with an annotated
  violation and 25 without, seed 42) and writes `items.json` + images.
- The images are uploaded to the artifact's asset store; `urls.json` maps
  image id -> asset url. `build.py` fills `page.tpl.html` into `audit.html`.
- Answers live in the artifact's database at
  `audit/<auditor id>/images/<image_id>`: `blind` (yes/no/unsure per rule,
  given before the annotation is shown and then locked), `boxes`
  (yes/partly/no: does each annotated box sit on its hazard), `note`.
  Each auditor writes only their own subtree; the owner reads all.

## First reader: Claude Opus 5.5 (decided 2026-11-11)

With no human auditor available, Claude answered the audit blind (seeded random
order, the dataset's rule text, a reason per image), committed those answers
(`7b4adeb`), then judged box placement after the reveal. Answers are in
`results/audit_answers/audit/claude-opus-5-5/` and the score in
`results/audit_score_claude-opus-5-5.txt`.

The owner then reviews every answer on a second page,
https://claude.ai/artifact/UqngRF6adtJNJ6aQCoeSNr, built by
`python audit/build_review.py OUT_DIR --img <Stage 12 export>/` from
`review.tpl.html`. Each saved review is `review/<image_id>` in that page's
database: `final` (yes/no/unsure per applicable rule), `note`, `claude` (the
blind answers it was set against), `reviewed_at`.

## Scoring

Export the answers, then score:

    # ArtifactData list, collection audit/<uid>/images, out_dir=results/audit_answers
    python scripts/13_audit_score.py results/audit_answers

Per auditor: "marked, not confirmed" (annotation marks a violation, auditor says
No) and "unmarked, auditor sees one" kept apart; "can't tell" on its own line;
only applicable rules. The image-level rate is reweighted from the 50/50 draw
to each tier's real hazard prevalence and the tiers' real shares
(`results/audit_strata.csv`: rich 771 of 5,271, poor 507 of 4,742). Per tier,
"marked, not confirmed" rests on ~25 images and sits below the n=30 floor by
design; quote it pooled across tiers.
