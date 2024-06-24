import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr

base_evaluation_dir = Path("./results/evaluation/")
base_likelihood_dir = Path("./results/likelihood/")
base_cor_dir = Path("./results/correlation/")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    eval_result_dir = base_evaluation_dir / args.model_path / args.data_name
    likelihood_result_dir = base_likelihood_dir / args.model_path / args.data_name

    # load data
    criteria_list = [criteria.name for criteria in (base_evaluation_dir / args.model_path / args.data_name).iterdir() if criteria.is_dir()]
    likelihood_score_list = pd.read_csv(likelihood_result_dir / "likelihood.csv")["likelihood"].to_list()

    result = pd.DataFrame(
        columns=[
            "Evaluation Performance(Pearson's r)",
            "Evaluation Performance(Spearman's rho)",
            "Evaluation Performance(Kendall's tau)",
            "Bias Score(Spearman's rho)",
            "Bias Score(Kendall's tau)",
        ]
    )

    average_evaluator_score_list = [0] * len(likelihood_score_list)
    average_human_score_list = [0] * len(likelihood_score_list)

    # Calculate correlation for each criteria
    for criteria in criteria_list:
        eval_path = eval_result_dir / criteria / args.method / "evaluation.csv"
        eval_data = pd.read_csv(eval_path)
        evaluator_score_list = eval_data["evaluation_score"].to_list()
        human_score_list = eval_data["human_score"].to_list()
        diff_score_list = [evaluator_score - human_score for evaluator_score, human_score in zip(evaluator_score_list, human_score_list)]
        average_evaluator_score_list = [evaluator_score + average_evaluator_score for evaluator_score, average_evaluator_score in zip(evaluator_score_list, average_evaluator_score_list)]
        average_human_score_list = [human_score + average_human_score for human_score, average_human_score in zip(human_score_list, average_human_score_list)]

        print(f"Criteria: {criteria}")
        print(f"Evaluation Performance(Pearson's rho): {pearsonr(evaluator_score_list, human_score_list)[0]}")
        print(f"Bias Score(Spearman's rho): {spearmanr(diff_score_list, likelihood_score_list).correlation}")
        line = [
            pearsonr(evaluator_score_list, human_score_list).correlation,
            spearmanr(evaluator_score_list, human_score_list).correlation,
            kendalltau(evaluator_score_list, human_score_list).correlation,
            spearmanr(diff_score_list, likelihood_score_list).correlation,
            kendalltau(diff_score_list, likelihood_score_list).correlation,
        ]
        result.loc[criteria] = line

    # Calculate average correlation
    average_evaluator_score_list = [average_evaluator_score / len(criteria_list) for average_evaluator_score in average_evaluator_score_list]
    average_human_score_list = [average_human_score / len(criteria_list) for average_human_score in average_human_score_list]
    average_diff_score_list = [evaluator_score - human_score for evaluator_score, human_score in zip(average_evaluator_score_list, average_human_score_list)]
    print("Average")
    print(f"Evaluation Performance(Pearson's rho): {pearsonr(average_evaluator_score_list, average_human_score_list)[0]}")
    print(f"Bias Score(Spearman's rho): {spearmanr(average_diff_score_list, likelihood_score_list).correlation}")
    line = [
        pearsonr(average_evaluator_score_list, average_human_score_list).correlation,
        spearmanr(average_evaluator_score_list, average_human_score_list).correlation,
        kendalltau(average_evaluator_score_list, average_human_score_list).correlation,
        spearmanr(average_diff_score_list, likelihood_score_list).correlation,
        kendalltau(average_diff_score_list, likelihood_score_list).correlation,
    ]
    result.loc["Average"] = line

    output_path = base_cor_dir / args.model_path / args.data_name / args.method / "correlation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)


if __name__ == "__main__":
    main()
