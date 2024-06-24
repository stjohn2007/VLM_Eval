from pathlib import Path

from src.utils.utils import format_generation_mm_vet

format_generation_mm_vet(Path("./raw_data/generation/mm-vet"), Path("./data/generation/mm-vet_long"))
