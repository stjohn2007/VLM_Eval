from pathlib import Path


class PromptMaker:
    def __init__(self, prompt_dir: Path):
        self.prompt_dir = prompt_dir
        self.system_message = self._load_system_message()
        self.input_text = self._load_input_text()

    def _load_system_message(self) -> str:
        with open(self.prompt_dir / "base.txt", "r") as f:
            text = f.read()
        text = text.strip()
        return text

    def _load_input_text(self) -> str:
        with open(self.prompt_dir / "input.txt", "r") as f:
            text = f.read()
        text = text.strip()
        return text

    def _make_input_prompt(self, target: str) -> str:
        return self.input_text.replace("{{input}}", target).strip()

    def make_prompt(self, target: str):
        return (self.system_message + "\n" + self._make_input_prompt(target)).strip()
