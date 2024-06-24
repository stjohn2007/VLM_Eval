from pathlib import Path

import hydra
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm

from src.utils.models import load_models
from src.utils.prompt import PromptMaker
from src.generation import generate_text


@hydra.main(config_path="./configs/generation/", config_name="config.yaml", version_base=None)
def main(cfg):
    result_dir = Path(cfg.base_path.result_dir) / cfg.name
    result_dir.mkdir(parents=True, exist_ok=True)
    print("Results will be saved in:", result_dir)
    with open(result_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # load model
    model = load_models(model_path=cfg.model.path, model_type=cfg.model.type)
    model.to("cuda")

    # generation
    output_path = result_dir / "generation.csv"

    prompt_dir = Path(cfg.base_path.prompt_dir) / cfg.prompt.name
    prompt_maker = PromptMaker(prompt_dir)
    generation_result = pd.DataFrame(columns=["source", "generated_text", "image_path"])

    # load data
    data_path = Path(cfg.base_path.data_dir) / cfg.data.name / "data.csv"
    data = pd.read_csv(data_path)

    # if the output file already exists, skip generation
    if not cfg.save.replace and output_path.exists():
        print(f"Output file: {output_path}, already exists. Skip generation.")
        return

    # generate
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        source = row["source"]
        image_path = row["image_path"]
        prompt = prompt_maker.make_prompt(source)
        generated_text = generate_text(
            model=model,
            prompt=prompt,
            image_path=image_path,
            temperature=cfg.generation.temperature
        )
        generation_result.loc[idx] = [source, generated_text, image_path]

    # save result
    generation_result.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
