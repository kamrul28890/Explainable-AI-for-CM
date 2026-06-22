"""Florence-2 loading and task execution.

Florence-2's custom modeling code (loaded via trust_remote_code) unconditionally
imports flash_attn, which has no usable Windows wheel. We strip that import
before from_pretrained runs and force the sdpa attention path instead.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.dynamic_module_utils import get_imports

from xai_pilot.config import MODEL_ID


def _get_imports_without_flash_attn(filename):
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


@contextmanager
def _no_flash_attn():
    with patch(
        "transformers.dynamic_module_utils.get_imports",
        _get_imports_without_flash_attn,
    ):
        yield


def load_florence2(model_id: str = MODEL_ID, device: str | None = None):
    """Load Florence-2 + its processor, patched to run on Windows/sdpa."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    with _no_flash_attn():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=dtype,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    model.eval()
    return model, processor


def run_task(
    model,
    processor,
    image,
    task_token: str,
    text_input: str | None = None,
    max_new_tokens: int = 1024,
    num_beams: int = 3,
    do_sample: bool = False,
    temperature: float = 1.0,
):
    """Run one Florence-2 task token on one image.

    Returns (raw_text, parsed, mean_token_prob) where `parsed` is the
    structured output from processor.post_process_generation and
    mean_token_prob is the confidence proxy (mean per-step token probability
    of the chosen sequence).
    """
    prompt = task_token if text_input is None else task_token + text_input
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype)

    generate_kwargs = dict(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        output_scores=True,
        return_dict_in_generate=True,
    )
    if do_sample:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**generate_kwargs)

    sequences = out.sequences
    raw_text = processor.batch_decode(sequences, skip_special_tokens=False)[0]

    parsed = processor.post_process_generation(
        raw_text, task=task_token, image_size=(image.width, image.height)
    )

    mean_token_prob = _mean_token_probability(out)
    return raw_text, parsed, mean_token_prob


def _mean_token_probability(generate_output) -> float:
    """Mean probability of the chosen token at each generation step.

    With beam search, generate_output.scores holds pre-beam-selection logits
    for the whole beam at each step, not the chosen sequence's logits
    directly, so this is a proxy, not an exact sequence likelihood. Good
    enough as a relative confidence signal across runs on the same model.
    """
    scores = generate_output.scores
    if not scores:
        return float("nan")
    probs = []
    for step_logits in scores:
        step_probs = torch.softmax(step_logits.float(), dim=-1)
        probs.append(step_probs.max(dim=-1).values.mean().item())
    return sum(probs) / len(probs)
