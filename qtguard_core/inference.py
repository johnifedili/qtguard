from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from qtguard_core.prompts import build_prompt
from qtguard_core.schema import QTGuardOutput


def _best_device() -> torch.device:
    # Prefer Apple Silicon GPU (MPS) if available; otherwise CPU.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _best_dtype(device: torch.device) -> torch.dtype:
    # MPS generally prefers float16; CPU safest with float32.
    if device.type == "mps":
        return torch.float16
    return torch.float32


@lru_cache(maxsize=1)
def _load_model_and_processor(model_id: str):
    """
    Loads MedGemma model + processor once per process.
    Model IDs (examples):
      - google/medgemma-1.5-4b-it
      - google/medgemma-4b-it
    """
    device = _best_device()
    dtype = _best_dtype(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # device_map="auto" works best on CUDA. On MPS/CPU, load then .to(device).
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    return model, processor, device, dtype


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract first JSON object from a model response.
    MedGemma may return extra tokens; we strip to the first {...} block.
    """
    # Find a JSON-like object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    candidate = text[start : end + 1]

    # Remove trailing junk after last brace if any (already trimmed above)
    return json.loads(candidate)


def generate_qtguard_output(
    mini_chart: str,
    model_id: Optional[str] = None,
    max_new_tokens: int = 800,
    retries: int = 2,
) -> QTGuardOutput:
    """
    Uses MedGemma to generate a QTGuardOutput (validated by Pydantic schema).
    Retries with a stricter instruction if the model output isn't valid JSON.
    """
    model_id = model_id or os.getenv("QTGUARD_MODEL_ID", "google/medgemma-1.5-4b-it")

    model, processor, device, dtype = _load_model_and_processor(model_id)

    base_prompt = build_prompt(mini_chart)

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        prompt = base_prompt
        if attempt > 0:
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Output MUST be valid JSON only. No markdown. No commentary."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move tensors to device + dtype
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True).strip()

        try:
            data = _extract_json(decoded)
            # Validate against schema (Pydantic v2)
            return QTGuardOutput.model_validate(data)
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"MedGemma generation failed after retries: {last_error}")
