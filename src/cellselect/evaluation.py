"""Validation-only choices followed by evaluation of fixed models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .metrics import prediction_metrics, selection_metrics
from .model import build_model
from .study import file_hash, run_name, save_json
from .training import predict


def freeze_selection(data, study_root):
    root = Path(study_root)
    path = root / "selection.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["data_manifest_sha256"] != data.manifest_sha256 or saved[
            "study_sha256"
        ] != file_hash(root / "study.json"):
            raise ValueError("Frozen choices do not match this study")
        return saved
    spec = json.loads((root / "study.json").read_text(encoding="utf-8"))
    if spec["data_manifest_sha256"] != data.manifest_sha256:
        raise ValueError("Data differ from the training study")
    rows = data.rows("val")
    meta = data.metadata.iloc[rows]
    target = meta.target.to_numpy(dtype=np.float64)
    ids = meta.global_index.to_numpy()
    threshold = data.manifest["threshold"]["value"]
    candidates = []
    for config in spec["runs"]:
        folder = root / "runs" / run_name(config)
        complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        actual = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if {k: v for k, v in actual.items() if k != "device"} != config:
            raise ValueError(f"Run configuration changed: {folder.name}")
        if complete["manifest_sha256"] != data.manifest_sha256:
            raise ValueError(f"Run used different data: {folder.name}")
        if complete["steps"] != config["max_steps"]:
            raise ValueError(f"Incomplete training: {folder.name}")
        if [item["step"] for item in complete["checkpoints"]] != config["checkpoint_steps"]:
            raise ValueError(f"Incomplete checkpoint menu: {folder.name}")
        for state in complete["checkpoints"]:
            prediction_path = folder / state["predictions_file"]
            checkpoint_path = folder / state["checkpoint_file"]
            if (
                file_hash(prediction_path) != state["predictions_sha256"]
                or file_hash(checkpoint_path) != state["checkpoint_sha256"]
            ):
                raise ValueError(f"Changed training artifact: {folder.name}")
            with np.load(prediction_path, allow_pickle=False) as predictions:
                if not np.array_equal(predictions["global_index"], ids):
                    raise ValueError("Validation prediction IDs do not match")
                diagnostics = prediction_metrics(
                    target,
                    threshold,
                    mean=predictions["mean"] if "mean" in predictions.files else None,
                    hit_logit=predictions["hit_logit"]
                    if "hit_logit" in predictions.files
                    else None,
                )
                for purpose, metric in (("mean", "mean_marker"), ("purity", "purity")):
                    score = (
                        "mean"
                        if config["method"] == "mse"
                        or (config["method"] == "joint" and purpose == "mean")
                        else "hit_logit"
                    )
                    utility = selection_metrics(
                        target,
                        predictions[score],
                        ids,
                        threshold,
                        spec["settings"]["primary_fraction"],
                    )[metric]
                    candidates.append(
                        {
                            "train_fraction": config["fraction"],
                            "method": config["method"],
                            "weight": config["lambda_binary"],
                            "seed": config["seed"],
                            "step": state["step"],
                            "selected_for": purpose,
                            "score": score,
                            "validation_utility": utility,
                            **diagnostics,
                            "checkpoint": checkpoint_path.relative_to(root).as_posix(),
                            "checkpoint_sha256": state["checkpoint_sha256"],
                        }
                    )
    frame = pd.DataFrame(candidates)
    selections = []
    for fraction in spec["settings"]["train_fractions"]:
        for purpose in ("mean", "purity"):
            for method in ("mse", "bce", "joint"):
                part = frame[
                    (frame.train_fraction == fraction)
                    & (frame.selected_for == purpose)
                    & (frame.method == method)
                ]
                menu = (
                    part.groupby(["weight", "step"])
                    .validation_utility.agg(["mean", "count"])
                    .reset_index()
                )
                if not (menu["count"] == len(spec["settings"]["seeds"])).all():
                    raise ValueError("A candidate is missing a paired seed")
                winner = menu.sort_values(
                    ["mean", "step", "weight"], ascending=[False, True, True]
                ).iloc[0]
                chosen = part[
                    (part.weight == winner.weight) & (part.step == winner.step)
                ].sort_values("seed")
                if chosen.seed.tolist() != spec["settings"]["seeds"]:
                    raise ValueError("Unexpected seed pairing")
                selections.append(
                    {
                        "train_fraction": fraction,
                        "method": method,
                        "selected_for": purpose,
                        "weight": float(winner.weight),
                        "step": int(winner.step),
                        "score": chosen.iloc[0].score,
                        "validation_utility": float(winner["mean"]),
                        "states": chosen[["seed", "checkpoint", "checkpoint_sha256"]].to_dict(
                            "records"
                        ),
                    }
                )
    result = {
        "data_manifest_sha256": data.manifest_sha256,
        "study_sha256": file_hash(root / "study.json"),
        "threshold": threshold,
        "primary_fraction": spec["settings"]["primary_fraction"],
        "selections": selections,
    }
    frame.to_csv(root / "validation_candidates.csv", index=False)
    save_json(path, result)
    return result


def evaluate_pool(data, study_root, device="cuda", pool="test"):
    if pool not in {"test", "external"}:
        raise ValueError("Choose the main test pool or the external pool")
    root = Path(study_root)
    selection_path = root / "selection.json"
    if not selection_path.exists():
        raise ValueError("Freeze validation choices before evaluating the test pool")
    choices = json.loads(selection_path.read_text(encoding="utf-8"))
    spec = json.loads((root / "study.json").read_text(encoding="utf-8"))
    if choices["data_manifest_sha256"] != data.manifest_sha256 or choices[
        "study_sha256"
    ] != file_hash(root / "study.json"):
        raise ValueError("Data or study changed after model selection")
    output = root / pool
    if output.exists():
        raise FileExistsError(
            "Test output already exists; preserve it and use the reporting command"
        )
    output.mkdir()
    save_json(
        output / "evaluation.json",
        {
            "selection_sha256": file_hash(selection_path),
            "data_manifest_sha256": data.manifest_sha256,
            "device": device,
            "status": "running",
        },
    )
    indices = data.rows(pool)
    meta = data.metadata.iloc[indices].reset_index(drop=True)
    ids = meta.global_index.to_numpy()
    target = meta.target.to_numpy(dtype=np.float64)
    images = np.asarray(data.images[indices], dtype=np.float32)
    threshold = choices["threshold"]
    states = {
        state["checkpoint"]: state for choice in choices["selections"] for state in choice["states"]
    }
    predictions = {}
    prediction_files = []
    torch.set_num_threads(spec["settings"]["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for i, (relative, state) in enumerate(states.items(), 1):
        checkpoint_path = root / relative
        if file_hash(checkpoint_path) != state["checkpoint_sha256"]:
            raise ValueError("A selected checkpoint changed")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            checkpoint["manifest_sha256"] != data.manifest_sha256
            or checkpoint["threshold"] != threshold
        ):
            raise ValueError("Checkpoint data binding differs")
        model = build_model().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        result = predict(model, images, checkpoint["normalization"], device, batch_size=512)
        method = checkpoint["config"]["method"]
        if method == "mse":
            result.pop("hit_logit")
        elif method == "bce":
            result.pop("mean")
        filename = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}.npz"
        np.savez_compressed(output / filename, global_index=ids, **result)
        prediction_files.append(
            {"checkpoint": relative, "file": filename, "sha256": file_hash(output / filename)}
        )
        predictions[relative] = result
        print(f"Test inference {i}/{len(states)}: {checkpoint_path.parent.name}", flush=True)
        del model
    metrics = []
    for choice in choices["selections"]:
        for state in choice["states"]:
            result = predictions[state["checkpoint"]]
            diagnostics = prediction_metrics(target, threshold, **result)
            for score_name, scores in result.items():
                for fraction in spec["settings"]["selection_fractions"]:
                    values = selection_metrics(target, scores, ids, threshold, fraction)
                    metrics.append(
                        {
                            "train_fraction": choice["train_fraction"],
                            "selected_for": choice["selected_for"],
                            "method": choice["method"],
                            "weight": choice["weight"],
                            "step": choice["step"],
                            "seed": state["seed"],
                            "score": score_name,
                            "primary_score": score_name == choice["score"],
                            "selection_fraction": fraction,
                            **values,
                            **diagnostics,
                        }
                    )
    meta[["global_index", "fov_id", "slide_id", "target"]].to_csv(output / "cells.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    save_json(
        output / "evaluation.json",
        {
            "selection_sha256": file_hash(selection_path),
            "data_manifest_sha256": data.manifest_sha256,
            "device": device,
            "status": "complete",
            "n_cells": len(meta),
            "n_fields": int(meta.fov_id.nunique()),
            "prediction_files": prediction_files,
            "cells_sha256": file_hash(output / "cells.csv"),
            "metrics_sha256": file_hash(output / "metrics.csv"),
        },
    )
    return pd.DataFrame(metrics)
