"""Train the declared runs; test evaluation is a separate step."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cellselect.training import PreparedData, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text())
    common = spec["common"]
    data, previous_data_key = None, None
    for run in spec["runs"]:
        config = {**common, **run}
        name = config.pop("name")
        # Per-run subsets are paired across objectives and nested within a seed.
        data_key = (
            config.get("marker", "CD16"),
            config.get("train_fraction", 1.0),
            config.get("subset_seed", 20261021),
        )
        if data_key != previous_data_key:
            data = PreparedData(
                args.data, marker=data_key[0], train_fraction=data_key[1], subset_seed=data_key[2]
            )
            previous_data_key = data_key
        run_training(data, config, Path(args.output) / name)


if __name__ == "__main__":
    main()
