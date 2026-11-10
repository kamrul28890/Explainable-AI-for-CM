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
