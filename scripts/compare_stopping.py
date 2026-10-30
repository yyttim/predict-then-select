"""Compare the two frozen stopping rules using paired fields and the same cells."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cellselect.metrics import paired_field_bootstrap, top_indices
from cellselect.study import file_hash


def load_scores(root, pool):
    selection = json.loads((root / "selection.json").read_text())
    evaluation = json.loads((root / pool / "evaluation.json").read_text())
    if evaluation["selection_sha256"] != file_hash(root / "selection.json"):
        raise ValueError("Selection binding differs")
    cells = pd.read_csv(root / pool / "cells.csv", float_precision="round_trip")
    if evaluation["cells_sha256"] != file_hash(root / pool / "cells.csv"):
        raise ValueError("Cell binding differs")
    files = {row["checkpoint"]: row for row in evaluation["prediction_files"]}
    scores = {}
    for choice in selection["selections"]:
        rows = []
        for state in choice["states"]:
            entry = files[state["checkpoint"]]
            path = root / pool / entry["file"]
            if file_hash(path) != entry["sha256"]:
                raise ValueError("Saved predictions changed")
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved["global_index"], cells.global_index.to_numpy()):
                    raise ValueError("Prediction identities differ")
                rows.append(saved[choice["score"]])
        scores[(choice["train_fraction"], choice["selected_for"], choice["method"])] = np.stack(
            rows
        )
    return selection, cells, scores


def compare(study, pool, output):
    root = Path(study)
    primary, cells, first = load_scores(root, pool)
    alternate, other_cells, second = load_scores(root / "stopping", pool)
    if not cells.equals(other_cells) or primary["threshold"] != alternate["threshold"]:
        raise ValueError("Stopping rules must use identical outcomes and threshold")
    spec = json.loads((root / "study.json").read_text())
    settings = spec["settings"]
    y, ids = cells.target.to_numpy(), cells.global_index.to_numpy()
    high = y >= primary["threshold"]
    differences, exchanges = [], []
    for key, scores in first.items():
        fraction, purpose, method = key
        rows = paired_field_bootstrap(
            y,
            second[key],
            scores,
            ids,
            cells.fov_id.to_numpy(),
            primary["threshold"],
            fractions=settings["selection_fractions"],
            draws=settings["bootstrap_draws"],
            seed=settings["bootstrap_seed"],
        )
        differences.extend(
            {
                "train_fraction": fraction,
                "selected_for": purpose,
                "method": method,
                "comparison": "predictive_minus_utility_stopping",
                **row,
            }
            for row in rows
        )
        for seed, a, b in zip(settings["seeds"], second[key], scores):
            for q in settings["selection_fractions"]:
                sa, sb = top_indices(a, ids, q), top_indices(b, ids, q)
                added, removed = np.setdiff1d(sa, sb), np.setdiff1d(sb, sa)
                assert len(added) == len(removed)
                exchanges.append(
                    {
                        "train_fraction": fraction,
                        "selected_for": purpose,
                        "method": method,
                        "seed": seed,
                        "selection_fraction": q,
                        "n_selected": len(sa),
                        "n_exchanged": len(added),
                        "high_added": int(high[added].sum()),
                        "high_removed": int(high[removed].sum()),
                        "mean_added": float(y[added].mean()) if len(added) else np.nan,
                        "mean_removed": float(y[removed].mean()) if len(removed) else np.nan,
                    }
                )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(differences).to_csv(output / "stopping_differences.csv", index=False)
    pd.DataFrame(exchanges).to_csv(output / "stopping_exchanges.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/study"))
    parser.add_argument("--pool", choices=("test", "external"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or (
        Path("results/stopping/tables")
        if args.pool == "test"
        else Path("results/stopping/external/tables")
    )
    compare(args.study, args.pool, destination)
