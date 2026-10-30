"""Paired FP32 training with fixed update budgets and validation checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import normalization
from .losses import objective_loss
from .model import build_model


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ShuffledBatches:
    """Cycle independent permutations, carrying leftovers into full-size batches."""

    def __init__(self, n, batch_size, seed):
        if n < 1 or batch_size < 1:
            raise ValueError("Positive data and batch sizes required")
        self.n, self.batch_size = n, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.position = torch.empty(0, dtype=torch.long), 0

    def __next__(self):
        pieces, needed = [], self.batch_size
        while needed:
            if self.position == len(self.order):
                self.order = torch.randperm(self.n, generator=self.generator)
                self.position = 0
            count = min(needed, self.n - self.position)
            pieces.append(self.order[self.position : self.position + count])
            self.position += count
            needed -= count
        return torch.cat(pieces)


def learning_rate(step, config):
    """Learning rate used for a one-based optimizer update."""
    total, warmup = config["max_steps"], config["warmup_steps"]
    if not 1 <= step <= total:
        raise ValueError("Step outside the training schedule")
    if warmup and step <= warmup:
        factor = step / warmup
    else:
        decay = total - warmup
        progress = (step - warmup - 1) / (decay - 1) if decay > 1 else 1.0
        factor = (
            config["min_lr_ratio"]
            + (1 - config["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2
        )
    return config["lr"] * factor


@torch.inference_mode()
def predict(model, images, normalization, device, batch_size=512):
    """Return real-unit means and event logits, both float64 arrays."""
    if not len(images) or batch_size < 1:
        raise ValueError("Nonempty images and positive batch size required")
    device = torch.device(device)
    model.eval()
    mean = torch.tensor(normalization["image_mean"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    std = torch.tensor(normalization["image_std"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    means, logits = [], []
    for start in range(0, len(images), batch_size):
        part = images[start : start + batch_size]
        x = (
            part.to(device=device, dtype=torch.float32)
            if isinstance(part, torch.Tensor)
            else torch.from_numpy(np.array(part, dtype=np.float32, copy=True)).to(device)
        )
        with torch.autocast(device_type=device.type, enabled=False):
            outputs = model((x - mean) / std)
        means.append(outputs["mean_z"].float().cpu().numpy())
        logits.append(outputs["hit_logit"].float().cpu().numpy())
    return {
        "mean": np.concatenate(means).astype(np.float64) * normalization["target_std"]
        + normalization["target_mean"],
        "hit_logit": np.concatenate(logits).astype(np.float64),
    }


def _validate(config):
    for key in ("seed", "max_steps", "batch_size", "warmup_steps", "cpu_threads"):
        value = config[key]
        minimum = 0 if key in {"seed", "warmup_steps"} else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {key}")
    steps = config["checkpoint_steps"]
    if (
        not isinstance(steps, list)
        or not steps
        or steps != sorted(set(steps))
        or any(
            isinstance(s, bool) or not isinstance(s, int) or not 1 <= s <= config["max_steps"]
            for s in steps
        )
        or steps[-1] != config["max_steps"]
    ):
        raise ValueError("checkpoint_steps must be sorted, unique, positive, and end at max_steps")
    if config["warmup_steps"] >= config["max_steps"] or config["seed"] >= 2**32:
        raise ValueError("Invalid warmup or seed")
    for key in ("lr", "weight_decay", "min_lr_ratio"):
        if not math.isfinite(config[key]):
            raise ValueError(f"Nonfinite {key}")
    if config["lr"] <= 0 or config["weight_decay"] < 0 or not 0 <= config["min_lr_ratio"] <= 1:
        raise ValueError("Invalid optimization settings")
    if config["method"] not in {"mse", "bce", "joint"} or config["fraction"] not in {0.25, 1.0}:
        raise ValueError("Invalid method or training fraction")
    weight = config.get("lambda_binary", 1.0)
    if not math.isfinite(weight) or weight < 0 or (config["method"] == "joint" and weight == 0):
        raise ValueError("lambda_binary must be nonnegative, and positive for joint training")
    json.dumps(config, allow_nan=False)


def train_run(data, config, output):
    """Train without test access; save every declared checkpoint and its validation predictions."""
    _validate(config)
    output = Path(output)
    manifest_sha256 = data.manifest_sha256
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise ValueError("Prepared data must expose its saved manifest SHA-256")
    if config.get("data_manifest_sha256", manifest_sha256) != manifest_sha256:
        raise ValueError("Configured data manifest differs")
    if (output / "complete.json").exists():
        done = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        if (
            json.loads((output / "config.json").read_text(encoding="utf-8")) != config
            or done["manifest_sha256"] != manifest_sha256
            or done["config_sha256"] != file_sha256(output / "config.json")
        ):
            raise ValueError("Completed run has different configuration or data")
        if (
            done["steps"] != config["max_steps"]
            or [candidate["step"] for candidate in done["checkpoints"]]
            != config["checkpoint_steps"]
        ):
            raise ValueError("Completed run has an incomplete checkpoint schedule")
        for candidate in done["checkpoints"]:
            for key in ("checkpoint", "predictions"):
                if file_sha256(output / candidate[key + "_file"]) != candidate[key + "_sha256"]:
                    raise ValueError("Completed run artifact changed")
        if (
            done["final_checkpoint_sha256"] != done["checkpoints"][-1]["checkpoint_sha256"]
            or done["final_predictions_sha256"] != done["checkpoints"][-1]["predictions_sha256"]
        ):
            raise ValueError("Completed final checkpoint binding differs")
        return done
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Partial run exists; use an empty output directory")
    train = np.asarray(data.rows("train", fraction=config["fraction"]))
    val = np.asarray(data.rows("val"))
    if not len(train) or not len(val) or np.intersect1d(train, val).size:
        raise ValueError("Training and validation rows must be nonempty and disjoint")
    norm = normalization(data, train)
    tau = float(data.manifest["threshold"]["value"])
    if not np.isfinite(tau):
        raise ValueError("Threshold must be finite")
    target = data.metadata.iloc[train].target.to_numpy(dtype=np.float64)
    y_val = data.metadata.iloc[val].target.to_numpy(dtype=np.float64)
    high = (target >= tau).astype(np.float32)
    if (
        not np.isfinite(target).all()
        or not np.isfinite(y_val).all()
        or set(np.unique(high)) != {0.0, 1.0}
    ):
        raise ValueError("Finite targets and both training event classes are required")
    tx = torch.from_numpy(np.array(data.images[train], dtype=np.float32, copy=True))
    vx = torch.from_numpy(np.array(data.images[val], dtype=np.float32, copy=True))
    if not torch.isfinite(tx).all() or not torch.isfinite(vx).all():
        raise ValueError("Nonfinite train/validation images")
    target_z = torch.tensor(
        (target - norm["target_mean"]) / norm["target_std"], dtype=torch.float32
    )
    high = torch.from_numpy(high)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    model = build_model(seed=seed).float().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    if (
        device.type == "cuda"
        and (tx.numel() + vx.numel()) * 4 < torch.cuda.mem_get_info()[0] * 0.25
    ):
        tx, target_z, high, vx = tx.to(device), target_z.to(device), high.to(device), vx.to(device)
    batches = ShuffledBatches(len(train), config["batch_size"], seed)
    image_mean = torch.tensor(norm["image_mean"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    image_std = torch.tensor(norm["image_std"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    checkpoints, started = [], time.perf_counter()
    fields = [
        "step",
        "learning_rate",
        "train_loss",
        "train_mse_z",
        "train_bce",
        "val_mse_z",
        "val_bce",
        "wall_seconds",
    ]
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for step in range(1, config["max_steps"] + 1):
            model.train()
            index = next(batches)
            lr = learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=False):
                outputs = model((tx[index].to(device) - image_mean) / image_std)
                loss, mse, bce = objective_loss(
                    outputs,
                    target_z[index].to(device),
                    high[index].to(device),
                    config["method"],
                    config.get("lambda_binary", 1.0),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            optimizer.step()
            row = {
                "step": step,
                "learning_rate": lr,
                "train_loss": float(loss.detach()),
                "train_mse_z": float(mse.detach()) if config["method"] != "bce" else None,
                "train_bce": float(bce.detach()) if config["method"] != "mse" else None,
            }
            if step in config["checkpoint_steps"]:
                predictions = predict(model, vx, norm, device, config["batch_size"])
                valid = {"global_index": data.metadata.iloc[val].global_index.to_numpy()}
                values = {}
                if config["method"] != "bce":
                    valid["mean"] = predictions["mean"]
                    values["mse_z"] = float(
                        np.mean(((predictions["mean"] - y_val) / norm["target_std"]) ** 2)
                    )
                if config["method"] != "mse":
                    valid["hit_logit"] = predictions["hit_logit"]
                    z = predictions["hit_logit"]
                    values["bce"] = float(np.mean(np.logaddexp(0.0, z) - (y_val >= tau) * z))
                if not all(np.isfinite(v).all() for v in valid.values()) or not all(
                    np.isfinite(v) for v in values.values()
                ):
                    raise FloatingPointError("Nonfinite validation output")
                cp, pred = f"step_{step:06d}.pt", f"val_{step:06d}.npz"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "normalization": norm,
                        "config": config,
                        "threshold": tau,
                        "step": step,
                        "manifest_sha256": manifest_sha256,
                    },
                    output / cp,
                )
                np.savez_compressed(output / pred, **valid)
                checkpoints.append(
                    {
                        "step": step,
                        "checkpoint_file": cp,
                        "checkpoint_sha256": file_sha256(output / cp),
                        "predictions_file": pred,
                        "predictions_sha256": file_sha256(output / pred),
                        "validation": values,
                    }
                )
                row.update({"val_" + key: value for key, value in values.items()})
            row["wall_seconds"] = time.perf_counter() - started
            writer.writerow(row)
            if step in config["checkpoint_steps"]:
                stream.flush()
                print(json.dumps({"run": output.name, **row}), flush=True)
    complete = {
        "steps": config["max_steps"],
        "wall_seconds": time.perf_counter() - started,
        "training_cells": len(train),
        "validation_cells": len(val),
        "examples_seen": config["max_steps"] * config["batch_size"],
        "manifest_sha256": manifest_sha256,
        "config_sha256": file_sha256(output / "config.json"),
        "final_checkpoint_sha256": checkpoints[-1]["checkpoint_sha256"],
        "final_predictions_sha256": checkpoints[-1]["predictions_sha256"],
        "checkpoints": checkpoints,
    }
    write_json(output / "complete.json", complete)
    return complete
