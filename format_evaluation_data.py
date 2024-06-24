from pathlib import Path

from src.utils.utils import format_evaluation_thumb_image_captioning

format_evaluation_thumb_image_captioning(Path("./raw_data/evaluation/thumb_image_captioning/mscoco_THumB-1.0.jsonl"), Path("./raw_data/evaluation/thumb_image_captioning/karpathy-test-split-images/"), Path("./data/evaluation/thumb_image_captioning"))
