# Study 2 — VLM Explanation Quality

Measuring whether a vision-language model's *explanation* of a construction-site
safety judgement can be trusted — not just whether its answer is right.

**Design:** [../../docs/architecture/vlm-xai-study-architecture.md](../../docs/architecture/vlm-xai-study-architecture.md)
**Running it:** [../../docs/setup/REPRODUCE.md](../../docs/setup/REPRODUCE.md)

---

## Status

| component | state |
|---|---|
| `rules.py` — the four rules as questions | ✅ implemented and tested |
| `parsing.py` — reply parsing and parse-rate tracking | ✅ implemented and tested |
| `sampling.py` — two-stratum sampling, prevalence reweighting | ✅ implemented and tested |
| `backends/base.py` — the `Backend` protocol | ✅ interface defined |
| `backends/qwen.py` — Qwen2.5-VL implementation | ⬜ not started (needs CUDA) |
| `metrics/` — the seven metrics | ⬜ not started |
| `scripts/` — stage runners | ⬜ skeletons only |

40 tests pass, none needing a GPU.

---

## What can be worked on without a GPU

Most of the interesting logic, as it happens:

```bash
pytest tests -q
```

- **Prompt wording** (`rules.py`). Changing a question and re-running the tests
  takes seconds. The exact phrasing materially affects results, and this is the
  cheapest place to iterate.
- **Reply parsing** (`parsing.py`). Every real model reply that fails to parse is a
  lost data point. Add it to `tests/test_parsing.py` as a case, then make it pass.
- **Sampling and reweighting** (`sampling.py`). Pure arithmetic over metadata.

Model inference needs CUDA — see
[SETUP.md § Which machine runs what](../../docs/setup/SETUP.md#which-machine-runs-what).

---

## Design decisions worth knowing before editing

**Every violated rule is asked.** The pilot kept only the highest-priority
violation and discarded the rest, so a photo with both a PPE breach and an
unguarded trench was only ever tested for PPE.

**Negative controls are context-aware.** The pilot assigned compliant images a rule
by blind rotation, which could ask about excavator proximity in a photo containing
no machinery — a question that cannot be failed, inflating accuracy while measuring
nothing. `rules_to_ask` only draws controls from *applicable* rules.

**The parser is tolerant but records tolerance.** A strict parser silently discards
understandable-but-loosely-formatted replies; a study losing 15% of its data to
formatting is reporting a metric over an unrepresentative subset. So variations are
accepted and flagged `strict=False`, and the non-strict share is reported. A rising
rate means the prompt is degrading.

**Sampling is two strata, not one balanced set.** Three of the seven metrics are
*undefined* on a NO answer — no region to mask, no rationale to grade — so compliant
images cannot contribute to them at all. They measure specificity and the
false-alarm rate instead, and the two strata are recombined by prevalence
reweighting.

Why that last point matters concretely, from `reweight_precision`:

```
sensitivity 0.80, specificity 0.90
  measured on a balanced 50/50 set  →  precision 0.89
  at the true 12.8% prevalence      →  precision 0.54
```

Reporting only the balanced figure overstates deployed precision by 35 points.

---

## Layout

```
src/xai_vlm/
    rules.py          four rules, prompt text, which-rules-to-ask policy
    parsing.py        reply → structured answer, with parse-rate tracking
    sampling.py       two-stratum construction, matching, reweighting
    backends/
        base.py       the Backend protocol and the Answer dataclass
    metrics/          the seven metrics (to come)
scripts/              numbered stage runners
tests/                40 tests, no GPU required
results/              output CSVs
```

---

## Next steps

1. **Stage 0 feasibility gate** on the CUDA machine — the study stops here if a
   quantised model cannot detect small hard hats. Stop rules are fixed in advance
   in the architecture document, §8 Stage 0.
2. **`backends/qwen.py`** implementing the protocol.
3. **Metrics**, in dependency order: descriptive accuracy → completeness →
   explanation correctness → the rest.
