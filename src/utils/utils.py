import requests
from io import BytesIO
from pathlib import Path
import json

import pandas as pd
from PIL import Image


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def format_evaluation_thumb_image_captioning(score_path: Path, image_path: Path, save_path: Path):
    aspects = [("precision", "P"), ("recall", "R")]
    score_range = [1, 2, 3, 4, 5]
    with open(score_path, "r") as f:
        scores = f.readlines()
        data = [json.loads(score) for score in scores]
    for aspect_name, key in aspects:
        aspect_save_dir = save_path / aspect_name / "data"
        aspect_save_dir.mkdir(parents=True, exist_ok=True)
        new_data = pd.DataFrame(columns=["source", "target", "score", "image_path"])
        new_data["source"] = ["Provide a one-sentence caption for the provided image."] * len(data)
        new_data["target"] = [d["hyp"] for d in data]
        new_data["score"] = [d[key] for d in data]
        new_data["image_path"] = [str(image_path / d["image"]) for d in data]
        new_data.to_csv(str(aspect_save_dir / "score.csv"), index=False)

        with open(aspect_save_dir / "score_range.txt", "w") as f:
            f.write("\n".join([str(score) for score in score_range]))
    return


def format_generation_mm_vet(data_path: Path, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    json_data = json.load(open((data_path / "mm-vet.json"), "r"))
    image_dir = data_path / "images"
    new_data = pd.DataFrame(columns=["source", "answer", "image_path", "original_key"])

    key_list = [f"v1_{idx}" for idx in range(138, 218)]
    for key in key_list:
        value = json_data[key]
        new_data.loc[len(new_data)] = [value["question"], value["answer"], str(image_dir / value["imagename"]), key]
    new_data.to_csv(save_path / "data.csv", index=False)
    return
