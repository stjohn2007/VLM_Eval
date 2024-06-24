from pathlib import Path

import hydra
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau, pearsonr

from src.utils.models import load_models
from src.utils.prompt import PromptMaker
from src.evaluation import calc_evaluation


@hydra.main(config_path="./configs/evaluation/", config_name="config.yaml", version_base=None)
def main(cfg):
    result_dir = Path(cfg.base_path.result_dir) / cfg.name
    result_dir.mkdir(parents=True, exist_ok=True)
    print(result_dir)
    with open(result_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    data_dir = Path(cfg.base_path.data_dir) / cfg.data.name

    # load model
    model = load_models(model_path=cfg.model.path, model_type=cfg.model.type)
    model.to("cuda")

    # evaluation
    criteria_list = [criteria.name for criteria in data_dir.iterdir() if criteria.is_dir()]
    for criteria_name in criteria_list:
        print(f"Processing {criteria_name}...")
        output_dir = result_dir / criteria_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "evaluation.csv"

        prompt_dir = Path(cfg.base_path.prompt_dir) / cfg.prompt.name / criteria_name
        prompt_maker = PromptMaker(prompt_dir)
        eval_result = pd.DataFrame(columns=["source", "evaluation_score", "human_score", "image_path"])

        # load data
        score_data = pd.read_csv(data_dir / criteria_name / "score.csv")
        with open(data_dir / criteria_name / "score_range.txt", "r") as f:
            score_range = [int(score.strip()) for score in f.readlines()]

        # if the output file already exists, skip the calculation
        if not cfg.save.replace and output_path.exists():
            print(f"Output file: {output_path}, already exists. Skip the calculation.")
            continue

        # evaluate
        for idx, row in tqdm(score_data.iterrows(), total=len(score_data)):
            target = row["target"]
            image_path = row["image_path"]
            human_score = row["score"]
            prompt = prompt_maker.make_prompt(target)
            evaluation_score = calc_evaluation(
                model=model,
                prompt=prompt,
                scores=score_range,
                image_path=image_path,
                method=cfg.evaluation.method,
            )
            eval_result.loc[idx] = [target, evaluation_score, human_score, image_path]

        # save result
        eval_result.to_csv(output_path, index=False)

    # calc correlation
    cor_result = pd.DataFrame(
        columns=[
            "Evaluation Performance(Pearson's r)",
            "Evaluation Performance(Spearman's rho)",
            "Evaluation Performance(Kendall's tau)",
        ]
    )

    average_evaluator_score_list = [0] * len(score_data)
    average_human_score_list = [0] * len(score_data)

    for criteria_name in criteria_list:
        eval_path = result_dir / criteria_name / "evaluation.csv"
        eval_data = pd.read_csv(eval_path)
        evaluator_score_list = eval_data["evaluation_score"].to_list()
        human_score_list = eval_data["human_score"].to_list()
        average_evaluator_score_list = [
            evaluator_score + average_evaluator_score
            for evaluator_score, average_evaluator_score in zip(evaluator_score_list, average_evaluator_score_list)
        ]
        average_human_score_list = [
            human_score + average_human_score
            for human_score, average_human_score in zip(human_score_list, average_human_score_list)
        ]

        print(f"Criteria: {criteria_name}")
        print(f"Evaluation Performance(Pearson's rho): {pearsonr(evaluator_score_list, human_score_list)[0]}")
        line = [
            pearsonr(evaluator_score_list, human_score_list)[0],
            spearmanr(evaluator_score_list, human_score_list).correlation,
            kendalltau(evaluator_score_list, human_score_list).correlation,
        ]
        cor_result.loc[criteria_name] = line

    # Calculate average correlation
    average_evaluator_score_list = [
        average_evaluator_score / len(criteria_list) for average_evaluator_score in average_evaluator_score_list
    ]
    average_human_score_list = [
        average_human_score / len(criteria_list) for average_human_score in average_human_score_list
    ]
    print("Average")
    print(
        f"Evaluation Performance(Pearson's rho): {pearsonr(average_evaluator_score_list, average_human_score_list)[0]}"
    )
    line = [
        pearsonr(average_evaluator_score_list, average_human_score_list)[0],
        spearmanr(average_evaluator_score_list, average_human_score_list).correlation,
        kendalltau(average_evaluator_score_list, average_human_score_list).correlation,
    ]
    cor_result.loc["Average"] = line
    cor_result.to_csv(result_dir / "correlation.csv")


if __name__ == "__main__":
    main()
