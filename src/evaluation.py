from pathlib import Path
import re

import torch
import numpy as np
from scipy.special import softmax

from src.utils.models import VLModel
from src.utils.utils import load_image


def calc_evaluation_direct(model: VLModel, prompt: str, scores: "list[int]", image_path: Path, device="cuda") -> float:
    model.to(device)
    model.eval()

    image = load_image(image_path)
    image_tensor, image_size = model.encode_image(image)
    image_tensor = image_tensor.to(device)
    prompt = model.build_prompt(prompt, None)
    input_ids = model.encode_text(prompt)
    input_ids = input_ids.to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_size,
            do_sample=False,
            num_beams=1,
            max_new_tokens=10,
            use_cache=True,
        )
    output_text = model.decode_text(output_ids, skip_special_tokens=True)[0].strip()
    match = re.search(r"\d+", output_text)
    if match:
        score = int(match.group())
    else:
        score = 0
    if score not in scores:
        score = 0
    return score


def calc_evaluation_expectation(model: VLModel, prompt: str, scores: "list[int]", image_path: Path, device="cuda") -> float:
    model.to(device)
    model.eval()

    image = load_image(image_path)
    image_tensor, image_size = model.encode_image(image)
    image_tensor = image_tensor.to(device)
    prompt = model.build_prompt(prompt, None)
    input_ids = model.encode_text(prompt)
    input_ids = input_ids.to(device)

    # TODO: なぜかLLaVA1.5では "1" -> [29871, 29896] のように 29871が頭につくので、それをinput_idsの末尾にしておく。正しいのかは検証する必要がある
    input_ids = torch.cat([input_ids, torch.tensor([[29871]]).to(device)], dim=1)

    score_ids = [model.tokenizer.encode(str(score))[-1] for score in scores]
    score_weights = []

    with torch.inference_mode():
        outputs = model(
            input_ids,
            images=image_tensor,
            image_sizes=image_size,
            labels=None
        )
        for score_id in score_ids:
            score_weights.append(outputs.logits[0, -1, score_id].item())
    score_weights = np.array(score_weights)
    score_weights = softmax(score_weights)
    scores = np.array(scores)

    expected_score = float(np.sum(scores * score_weights))
    return expected_score


evaluation_methods = {
    "direct": calc_evaluation_direct,
    "expectation": calc_evaluation_expectation,
}


def calc_evaluation(model: VLModel, prompt: str, scores: "list[int]", image_path: Path, method: str, device="cuda") -> float:
    if method not in evaluation_methods:
        raise ValueError(f"Invalid method: {method}")
    return evaluation_methods[method](model, prompt, scores, image_path, device)
