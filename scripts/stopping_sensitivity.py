"""Freeze an exploratory proper-loss stopping comparison on the existing model bank."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from cellselect.study import file_hash, save_json


def freeze_stopping(study_root):
    root = Path(study_root)
    output = root / "stopping"
    if output.exists():
        raise FileExistsError("Stopping-rule choices already exist; preserve this analysis")
    spec = json.loads((root / "study.json").read_text(encoding="utf-8"))
    primary = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    if primary["study_sha256"] != file_hash(root / "study.json"):
        raise ValueError("Primary study binding changed")
    candidates = pd.read_csv(root / "validation_candidates.csv", float_precision="round_trip")
    selections = []
    for original in primary["selections"]:
        fraction, purpose, method = (
            original[k] for k in ("train_fraction", "selected_for", "method")
        )
        part = candidates[
            (candidates.train_fraction == fraction)
            & (candidates.selected_for == purpose)
            & (candidates.method == method)
        ].copy()
        continuous = method == "mse" or (method == "joint" and purpose == "mean")
        part["proper_loss"] = part.rmse**2 if continuous else part.log_loss
        if not np.isfinite(part.proper_loss).all():
            raise ValueError("The required predictive-head metric is missing")
        menu = part.groupby(["weight", "step"]).proper_loss.agg(["mean", "count"]).reset_index()
        if not menu["count"].eq(len(spec["settings"]["seeds"])).all():
            raise ValueError("Incomplete paired model bank")
        winner = menu.sort_values(["mean", "step", "weight"]).iloc[0]
        chosen = part[(part.weight == winner.weight) & (part.step == winner.step)].sort_values(
            "seed"
        )
        states = []
        for row in chosen.itertuples():
            if file_hash(root / row.checkpoint) != row.checkpoint_sha256:
                raise ValueError("A model checkpoint changed")
            states.append(
                {
                    "seed": int(row.seed),
                    "checkpoint": "../" + row.checkpoint,
                    "checkpoint_sha256": row.checkpoint_sha256,
                }
            )
        selections.append(
            {
                "train_fraction": fraction,
                "method": method,
                "selected_for": purpose,
                "weight": float(winner.weight),
                "step": int(winner.step),
                "score": original["score"],
                "validation_utility": float(chosen.validation_utility.mean()),
                "validation_loss": float(winner["mean"]),
                "loss": "mse" if continuous else "log_loss",
                "states": states,
            }
        )
    output.mkdir()
    shutil.copyfile(root / "study.json", output / "study.json")
    save_json(
        output / "selection.json",
        {
            "analysis": "Exploratory stopping-rule sensitivity added after the primary evaluation",
            "selection_rule": "Predictive loss of the deployment head; MSE uses MSE and BCE uses log loss",
            "data_manifest_sha256": primary["data_manifest_sha256"],
            "study_sha256": file_hash(output / "study.json"),
            "primary_selection_sha256": file_hash(root / "selection.json"),
            "validation_candidates_sha256": file_hash(root / "validation_candidates.csv"),
            "selection_source_sha256": file_hash(Path(__file__)),
            "threshold": primary["threshold"],
            "primary_fraction": primary["primary_fraction"],
            "selections": selections,
        },
    )
    return selections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/study"))
    args = parser.parse_args()
    for choice in freeze_stopping(args.study):
        print(
            f"{choice['train_fraction']:g} {choice['selected_for']} {choice['method']}: "
            f"{choice['loss']}, step {choice['step']}, weight {choice['weight']:g}"
        )
