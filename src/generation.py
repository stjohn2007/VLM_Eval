from pathlib import Path

import torch

from src.utils.models import VLModel
from src.utils.utils import load_image


def generate_text(model: VLModel, prompt: str, image_path: Path, temperature: float, device="cuda") -> str:
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
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=1024,
            use_cache=True,
        )
    output_text = model.decode_text(output_ids, skip_special_tokens=True)[0].strip()
    return output_text
