"""The four safety rules, as questions put to a vision-language model.

The pilot could not ask questions -- Florence-2 has no VQA head -- so rules were
encoded as box-distance heuristics, and `rule_1` collapsed to grounding the single
phrase "hard hat". Measured against the dataset's own violation reasons, 16.9% of
real PPE violations never mention a hard hat at all and were therefore undetectable
by construction.

Here each rule is simply the question, phrased to cover what the annotations
actually cover. See docs/architecture/vlm-xai-study-architecture.md section 6.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

RuleId = Literal["rule_1", "rule_2", "rule_3", "rule_4"]
RULE_IDS: tuple[RuleId, ...] = ("rule_1", "rule_2", "rule_3", "rule_4")


@dataclass(frozen=True)
class Rule:
    """One safety rule and everything needed to ask about it."""

    rule_id: RuleId
    hazard: str
    question: str
    # Dataset annotation fields that indicate this rule is *applicable* to an
    # image. Used to avoid vacuous questions (see should_ask_negative_control).
    context_fields: tuple[str, ...] = ()

    @property
    def violation_field(self) -> str:
        return f"{self.rule_id}_violation"


# The instruction block appended to every question. Content is constrained rather
# than length: models are unreliable at counting words, and "one sentence naming
# the specific person or location" is both enforceable and produces exactly the
# two things metric 7 needs -- a subject and a predicate.
RESPONSE_FORMAT = """
Answer in exactly this format, with nothing before or after:

ANSWER: YES or NO
REGION: (x0, y0, x1, y1) pixel coordinates of the single most relevant area, or NONE
REASON: One sentence naming the specific person or location and what is wrong.
""".strip()


RULES: dict[RuleId, Rule] = {
    "rule_1": Rule(
        rule_id="rule_1",
        hazard="ppe_violation",
        question=(
            "Look at every worker in this image. Is any worker missing required "
            "personal protective equipment - a hard hat, high-visibility clothing, "
            "or adequate body covering?"
        ),
    ),
    "rule_2": Rule(
        rule_id="rule_2",
        hazard="fall_hazard",
        question=(
            "Is any worker working at height without fall protection such as a "
            "safety harness or lanyard?"
        ),
    ),
    "rule_3": Rule(
        rule_id="rule_3",
        hazard="fall_hazard",
        question=(
            "Is there an unprotected edge, excavation, trench, or floor opening "
            "that a worker could fall into - one lacking guardrails or barriers?"
        ),
    ),
    "rule_4": Rule(
        rule_id="rule_4",
        hazard="struck_by_risk",
        question=(
            "Is any worker inside the danger zone or blind spot of operating heavy "
            "machinery, such as an excavator?"
        ),
        # Asking about excavator proximity in a photo with no machinery cannot be
        # failed, so it inflates accuracy without measuring anything.
        context_fields=("excavator",),
    ),
}


def prompt_for(rule_id: RuleId) -> str:
    """The complete text sent to the model for one rule."""
    return f"{RULES[rule_id].question}\n\n{RESPONSE_FORMAT}"


def violated_rules(row: dict) -> list[RuleId]:
    """Rules this dataset row actually violates, in fixed order."""
    return [r for r in RULE_IDS if row.get(RULES[r].violation_field) is not None]


def rule_is_applicable(rule_id: RuleId, row: dict) -> bool:
    """Whether the rule could meaningfully be failed by this image.

    A rule with no context requirement is always applicable. `rule_4` needs
    machinery present, otherwise the question is vacuous.
    """
    rule = RULES[rule_id]
    if not rule.context_fields:
        return True
    return any(row.get(field) for field in rule.context_fields)


def rules_to_ask(row: dict, seed: int = 0) -> list[tuple[RuleId, bool]]:
    """Which rules to ask of one image, as (rule_id, is_negative_control) pairs.

    Policy (architecture doc section 6.4), replacing the pilot's two defective
    behaviours:

    * The pilot kept only the single highest-priority violation, discarding every
      other hazard on multi-hazard images. Here **every violated rule is asked**.
    * The pilot assigned compliant images a rule by blind rotation, which could ask
      about excavators in a photo containing none. Here negative controls are drawn
      only from *applicable* rules.

    A negative control -- a rule the image does not violate -- is always included
    where one is applicable, so a model that simply answers YES to everything is
    detectable rather than scoring well.
    """
    rng = random.Random(f"{row.get('image_id', '')}-{seed}")
    violated = violated_rules(row)

    asked: list[tuple[RuleId, bool]] = [(r, False) for r in violated]

    candidates = [
        r for r in RULE_IDS
        if r not in violated and rule_is_applicable(r, row)
    ]
    if candidates:
        asked.append((rng.choice(candidates), True))
    elif not asked:
        # Neither violated nor applicable-negative: fall back to the rules that
        # need no context, so the image still contributes a specificity data point
        # rather than being silently dropped from the compliant stratum.
        contextless = [r for r in RULE_IDS if not RULES[r].context_fields]
        asked.append((rng.choice(contextless), True))

    return asked


def expected_answer(rule_id: RuleId, row: dict) -> Literal["YES", "NO"]:
    """Ground-truth answer for this rule on this image."""
    return "YES" if row.get(RULES[rule_id].violation_field) is not None else "NO"
