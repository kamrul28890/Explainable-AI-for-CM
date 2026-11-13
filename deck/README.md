# Presentation deck

A project-dissertation deck covering both studies, generated from committed
result files rather than assembled by hand.

```bash
python deck/build_assets.py   # figures + worked examples from results/*.csv
python deck/build_deck.py     # -> deck/Construction-VLM-XAI.pptx
```

Re-running both refreshes every number and chart. That is the point: the deck is
a view over the results, not a second copy of them that can drift.

| File | Purpose |
|---|---|
| `build_assets.py` | Charts and annotated worked examples (matplotlib + PIL) |
| `build_deck.py` | 42 slides (python-pptx), 16:9 |
| `assets/` | Generated PNGs — safe to delete, regenerable |
| `Construction-VLM-XAI.pptx` | The deck |

## Notes

- **Stage 4 is a placeholder** (slide 36) and reads its progress from the results
  file at build time. Rebuild once the run completes to fill in the flip rates.
- **Pipeline diagrams are native PowerPoint shapes**, so they can be edited in
  PowerPoint rather than regenerated here.
- **Worked-example photographs** come from `LouisChen15/ConstructionSite`
  (gated, CC-BY-NC-4.0) — fine for academic presentation, not for redistribution.
