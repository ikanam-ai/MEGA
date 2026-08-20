import subprocess
import sys
from pathlib import Path

DATASETS_ROOT = Path(__file__).parent.parent.parent.parent / "datasets"


def pull_tape() -> None:
    out = DATASETS_ROOT / "tape"
    if out.exists() and any(out.iterdir()):
        print(f"[tape] already present at {out}, skipping.")
        return
    print("[tape] Downloading from HuggingFace...")
    subprocess.run(
        [
            "huggingface-cli", "download",
            "--repo-type", "dataset",
            "RussianNLP/tape",
            "--local-dir", str(out),
            "--quiet",
        ],
        check=True,
    )
    print(f"[tape] Done → {out}")


def pull_sensitive_topics() -> None:
    out = DATASETS_ROOT / "russian_sensitive_inappropriate_topics"
    if out.exists() and any(out.iterdir()):
        print(f"[sensitive_topics] already present at {out}, skipping.")
        return
    print("[sensitive_topics] Cloning from GitHub...")
    subprocess.run(
        ["git", "clone", "https://github.com/s-nlp/inappropriate-sensitive-topics", str(out)],
        check=True,
    )
    print(f"[sensitive_topics] Done → {out}")


if __name__ == "__main__":
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
    pull_tape()
    pull_sensitive_topics()
    print("\nAll datasets ready.")
