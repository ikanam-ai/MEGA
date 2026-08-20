import argparse
from pathlib import Path

from valar.config import load_experiment_config
from valar.artifacts.io import write_jsonl


def build_tape_bank(cfg: dict, output_dir: Path) -> None:
    from valar.datasets.tape import iter_items, TAPE_SUBSETS

    dataset_cfg = cfg["dataset"]
    local_path = Path(dataset_cfg["local_path"])
    subsets = dataset_cfg.get("subsets", TAPE_SUBSETS)
    output_path = output_dir / "tape_annotation_items.jsonl"

    rows = []
    for subset in subsets:
        print(f"  Loading tape/{subset}...")
        try:
            for item in iter_items(local_path, subset):
                if item["text"].strip():
                    rows.append(item)
        except Exception as e:
            print(f"  WARNING: failed to load {subset}: {e}")

    write_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} items → {output_path}")


def build_sensitive_topics_bank(cfg: dict, output_dir: Path) -> None:
    from valar.datasets.sensitive_topics import iter_items

    dataset_cfg = cfg["dataset"]
    local_path = Path(dataset_cfg["local_path"])
    output_path = output_dir / "sensitive_topics_annotation_items.jsonl"

    rows = []
    for version in dataset_cfg.get("versions", ["Version1"]):
        for subset in dataset_cfg.get("subsets", ["appropriateness"]):
            print(f"  Loading {version}/{subset}...")
            try:
                for item in iter_items(local_path, version=version, subset=subset):
                    if item["text"].strip():
                        rows.append(item)
            except Exception as e:
                print(f"  WARNING: failed to load {version}/{subset}: {e}")

    write_jsonl(rows, output_path)
    print(f"Wrote {len(rows)} items → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output-dir", default="data/item_banks/valar/")
    args = parser.parse_args()

    cfg = load_experiment_config(args.experiment_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = cfg["experiment"]["dataset"]
    if dataset_id == "tape":
        build_tape_bank(cfg, output_dir)
    elif dataset_id == "russian_sensitive_inappropriate_topics":
        build_sensitive_topics_bank(cfg, output_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_id}")


if __name__ == "__main__":
    main()
