"""Matched training runs with validation-only monitoring and sealed test data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .metrics import prediction_metrics, selection_metrics
from .model import build_model


def objective_loss(scores, targets, logits, classes, config, epoch=None):
    """One shared training path for prediction and decision-aware objectives.

    Decision losses operate on a random training minibatch as a candidate pool.
    Validation and test always rank the entire declared held-out candidate pool.
    """
    mse = F.mse_loss(scores.float(), targets.float())
    ce = F.cross_entropy(logits.float(), classes)
    if config.get("mode", "joint") == "classifier":
        return ce, mse, ce
    objective = config.get("objective", "mse")
    if (
        objective == "pg_central"
        and epoch is not None
        and epoch <= config.get("decision_warmup_epochs", 0)
    ):
        objective = "spo_plus"
    if objective == "mse":
        primary = mse
    elif objective == "spo_plus":
        from .losses import topk_spo_plus

        primary = topk_spo_plus(
            scores.float(), targets.float(), fraction=float(config.get("primary_fraction", 0.1))
        )
    elif objective == "pg_central":
        from .losses import topk_pg_central

        primary = topk_pg_central(
            scores.float(),
            targets.float(),
            fraction=float(config.get("primary_fraction", 0.1)),
            h=float(config["pg_h"]),
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return primary + float(config.get("lambda_class", 0)) * ce, mse, ce


def decision_active_fraction(scores, targets, config, epoch):
    """Fraction of minibatch scores with a nonzero decision-loss subgradient."""
    objective = config.get("objective", "mse")
    if objective == "mse":
        return None
    scores, targets = scores.detach(), targets.detach()
    k = math.ceil(float(config.get("primary_fraction", 0.1)) * len(scores))
    if objective == "spo_plus" or epoch <= config.get("decision_warmup_epochs", 0):
        a, b = 2 * scores - targets, targets
    else:
        h = float(config["pg_h"])
        a, b = scores + h * targets, scores - h * targets
    masks = []
    for values in (a, b):
        chosen = torch.argsort(values, descending=True, stable=True)[:k]
        mask = torch.zeros_like(values, dtype=torch.bool)
        mask[chosen] = True
        masks.append(mask)
    return float((masks[0] != masks[1]).float().mean())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shared_gradient_diagnostic(model, images, marker, classes):
    """Measure task-gradient norms and cosine on a fixed training batch."""
    was_training = model.training
    model.eval()  # No dropout or BatchNorm-state changes during diagnostics.
    outputs = model(images)
    regression = F.mse_loss(outputs["marker"], marker)
    classification = F.cross_entropy(outputs["logits"], classes)
    parameters = tuple(model.encoder.parameters())
    rg = torch.autograd.grad(regression, parameters, retain_graph=True)
    cg = torch.autograd.grad(classification, parameters)
    rnorm = torch.sqrt(sum(g.square().sum() for g in rg))
    cnorm = torch.sqrt(sum(g.square().sum() for g in cg))
    dot = sum((a * b).sum() for a, b in zip(rg, cg))
    result = {
        "regression_norm": float(rnorm),
        "classification_norm": float(cnorm),
        "cosine": float(dot / (rnorm * cnorm)) if rnorm > 0 and cnorm > 0 else None,
        "batch_size": len(images),
        "partition": "train",
        "mode": "eval",
    }
    model.train(was_training)
    return result


class PreparedData:
    """Load a prepared cohort; training accesses only train/validation rows."""

    def __init__(self, directory, marker="CD16", train_fraction=1.0, subset_seed=20261021):
        directory = Path(directory)
        self.directory = directory
        self.manifest_hash = file_sha256(directory / "manifest.json")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for filename, expected in manifest.get("prepared_sha256", {}).items():
            if file_sha256(directory / filename) != expected:
                raise ValueError(f"Prepared data checksum mismatch: {filename}")
        self.metadata = pd.read_csv(directory / "metadata.csv")
        self.images = np.load(directory / "images.npy", mmap_mode="r")
        self.marker = marker
        self.train_fraction = float(train_fraction)
        self.subset_seed = int(subset_seed)
        if len(self.images) != len(self.metadata) or self.images.shape[1:] != (4, 28, 28):
            raise ValueError("Prepared images and metadata do not align")
        required = {"global_index", "class_id", marker, "fov_id", "split"}
        if not required.issubset(self.metadata.columns):
            raise ValueError(f"Missing metadata columns: {required - set(self.metadata.columns)}")
        if not self.metadata.global_index.is_unique:
            raise ValueError("Cell IDs must be unique")
        if self.metadata.groupby("fov_id")["split"].nunique().max() != 1:
            raise ValueError("Field leakage across partitions")
        self.group_column = "split_group" if "split_group" in self.metadata else "fov_id"
        if self.metadata.groupby(self.group_column)["split"].nunique().max() != 1:
            raise ValueError("Duplicate-linked fields leak across partitions")
        self.indices = {
            p: np.flatnonzero(self.metadata.split.to_numpy() == p) for p in ["train", "val", "test"]
        }
        if any(len(x) == 0 for x in self.indices.values()):
            raise ValueError("All three partitions must be nonempty")
        if not 0 < train_fraction <= 1:
            raise ValueError("Training fraction must be in (0,1]")
        # A shared group order gives nested subsets across training budgets.
        if train_fraction < 1:
            rows = self.indices["train"]
            groups = self.metadata.iloc[rows][self.group_column].to_numpy()
            order = np.random.default_rng(subset_seed).permutation(np.unique(groups))
            keep = order[: max(1, math.ceil(train_fraction * len(order)))]
            self.indices["train"] = rows[np.isin(groups, keep)]
        train = self.indices["train"]
        labels = self.metadata.iloc[train].class_id.to_numpy(dtype=np.int64)
        if set(labels) != {0, 1, 2}:
            raise ValueError("Training cohort must represent all three classes")
        target = self.metadata.iloc[train][marker].to_numpy(dtype=np.float64)
        if not np.isfinite(target).all() or np.std(target) <= 0:
            raise ValueError("Training marker values must be finite and variable")
        self.target_mean, self.target_std = float(target.mean()), float(target.std())
        self.class_means = np.array([target[labels == c].mean() for c in range(3)])
        # Streaming channel moments preserve archived amplitudes; no per-image scaling.
        total, sums, squares = 0, np.zeros(4), np.zeros(4)
        for start in range(0, len(train), 512):
            values = np.asarray(self.images[train[start : start + 512]], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite training images")
            total += values.shape[0] * values.shape[2] * values.shape[3]
            sums += values.sum(axis=(0, 2, 3))
            squares += (values**2).sum(axis=(0, 2, 3))
        self.image_mean = sums / total
        self.image_std = np.sqrt(np.maximum(squares / total - self.image_mean**2, 0))
        if (self.image_std <= 0).any():
            raise ValueError("Constant image channel")

    def normalization(self):
        return {
            "marker": self.marker,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "class_means": self.class_means.tolist(),
            "image_mean": self.image_mean.tolist(),
            "image_std": self.image_std.tolist(),
        }

    def partition(self, split):
        index = self.indices[split]
        meta = self.metadata.iloc[index]
        x = torch.from_numpy(np.asarray(self.images[index], dtype=np.float32).copy())
        y = torch.tensor(
            (meta[self.marker].to_numpy() - self.target_mean) / self.target_std, dtype=torch.float32
        )
        c = torch.tensor(meta.class_id.to_numpy(), dtype=torch.long)
        return x, y, c, meta


@torch.inference_mode()
def predict(model, images, mean, std, device, batch_size=512, amp=False):
    model.eval()
    logits, marker = [], []
    for start in range(0, len(images), batch_size):
        x = images[start : start + batch_size].to(device, non_blocking=True)
        x = (x - mean) / std
        # Keep inference FP32 even when training used AMP: ranking is tie-sensitive.
        with torch.autocast(device_type=device.type, enabled=False):
            out = model(x)
        logits.append(out["logits"].float().cpu().numpy())
        marker.append(out["marker"].float().cpu().numpy())
    return np.concatenate(logits), np.concatenate(marker)


def run_training(data: PreparedData, config: dict, output):
    """Train a fixed schedule and retain declared checkpoints without test access."""
    for key in ["epochs", "batch_size", "eval_every"]:
        value = config.get(key, 5 if key == "eval_every" else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    candidate_epochs = config.get("checkpoint_epochs", [config["epochs"]])
    if (
        not isinstance(candidate_epochs, list)
        or not candidate_epochs
        or any(
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or not 1 <= epoch <= config["epochs"]
            for epoch in candidate_epochs
        )
        or candidate_epochs != sorted(set(candidate_epochs))
    ):
        raise ValueError(
            "checkpoint_epochs must be a sorted, nonempty list of unique positive epochs within the training schedule"
        )
    reference_epoch = config.get("reference_epoch")
    if reference_epoch is not None and (
        isinstance(reference_epoch, bool)
        or not isinstance(reference_epoch, int)
        or reference_epoch not in candidate_epochs
    ):
        raise ValueError("reference_epoch must be an integer included in checkpoint_epochs")
    for key, default in [("loss_scale", 1.0), ("learning_rate", None)]:
        value = config.get(key, default)
        if value is None or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not np.isfinite(config.get("lambda_class", 0.0)) or config.get("lambda_class", 0.0) < 0:
        raise ValueError("lambda_class must be finite and nonnegative")
    if config.get("mode", "joint") == "classifier" and config.get("lambda_class", 0.0) != 0:
        raise ValueError("Classifier mode uses CE only; lambda_class must be zero")
    objective = config.get("objective", "mse")
    if objective not in {"mse", "spo_plus", "pg_central"}:
        raise ValueError("Unknown training objective")
    if config.get("mode", "joint") == "classifier" and objective != "mse":
        raise ValueError("Classifier mode cannot use a marker decision loss")
    if objective == "pg_central" and (
        not np.isfinite(config.get("pg_h", np.nan)) or config["pg_h"] <= 0
    ):
        raise ValueError("pg_h must be finite and positive")
    if objective != "pg_central" and "pg_h" in config:
        raise ValueError("pg_h applies only to pg_central")
    decision_warmup = config.get("decision_warmup_epochs", 0)
    if (
        isinstance(decision_warmup, bool)
        or not isinstance(decision_warmup, int)
        or not 0 <= decision_warmup < config["epochs"]
        or (objective != "pg_central" and decision_warmup)
    ):
        raise ValueError("Decision warmup must be a valid PG-only prefix of the training schedule")
    if objective == "pg_central" and any(e <= decision_warmup for e in candidate_epochs):
        raise ValueError("PG candidates must occur after the SPO+ warmup")
    for key, actual, default in [
        ("marker", data.marker, "CD16"),
        ("train_fraction", data.train_fraction, 1.0),
        ("subset_seed", data.subset_seed, 20261021),
    ]:
        if config.get(key, default) != actual:
            raise ValueError(f"Configured {key} does not match prepared training data")
    output = Path(output)
    if (output / "complete.json").exists():
        saved = json.loads((output / "config.json").read_text())
        if saved != config:
            raise ValueError("Existing completed run has a different configuration")
        previous = json.loads((output / "complete.json").read_text())
        provenance = json.loads((output / "provenance.json").read_text())
        train_ids = data.metadata.iloc[data.indices["train"]].global_index.to_numpy()
        current_ids = hashlib.sha256(train_ids.tobytes()).hexdigest()
        for name in ["model.py", "training.py", "metrics.py", "data.py"] + (
            ["losses.py"] if objective != "mse" else []
        ):
            if provenance.get("source_sha256", {}).get(name) != file_sha256(
                Path(__file__).parent / name
            ):
                raise ValueError(f"Completed run used a different implementation: {name}")
        if (
            provenance["manifest_sha256"] != data.manifest_hash
            or provenance["train_ids_sha256"] != current_ids
            or json.loads((output / "normalization.json").read_text()) != data.normalization()
            or file_sha256(output / "final.pt") != previous["checkpoint_sha256"]
        ):
            raise ValueError("Completed run provenance does not match current data/checkpoint")
        candidates = previous.get("candidates", [])
        if [candidate.get("epoch") for candidate in candidates] != candidate_epochs:
            raise ValueError("Completed run has a different checkpoint candidate grid")
        for candidate in candidates:
            epoch = candidate["epoch"]
            for path_key, hash_key, filename in [
                ("checkpoint_file", "checkpoint_sha256", f"snapshots/epoch_{epoch:03d}.pt"),
                (
                    "predictions_file",
                    "predictions_sha256",
                    f"snapshots/epoch_{epoch:03d}_validation.npz",
                ),
            ]:
                path = output / filename
                if (
                    candidate.get(path_key) != filename
                    or not path.is_file()
                    or file_sha256(path) != candidate.get(hash_key)
                ):
                    raise ValueError(f"Completed checkpoint candidate artifact changed: {filename}")
        return previous
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Partial run present; use a new explicit output path: {output}")
    seed = int(config["seed"])
    seed_everything(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(config.get("cpu_threads", 8)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(config.get("amp", False))
    model = build_model(seed=seed, feature_dim=int(config.get("feature_dim", 256))).to(device)
    mode = config.get("mode", "joint")
    lam = float(config.get("lambda_class", 0.0))
    if mode not in {"joint", "classifier"} or lam < 0:
        raise ValueError("Invalid training mode or classification weight")
    output.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epochs, batch_size = int(config["epochs"]), int(config["batch_size"])
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    warmup = int(config.get("warmup_epochs", 5))
    minimum = float(config.get("min_lr_ratio", 0.05))

    def lr_factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    train_x, train_y, train_c, train_meta = data.partition("train")
    val_x, _, val_c, val_meta = data.partition("val")
    mean = torch.tensor(data.image_mean, dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.tensor(data.image_std, dtype=torch.float32, device=device)[None, :, None, None]
    # Cache the full train/validation tensors when GPU memory permits.
    cache = config.get("cache_on_gpu", True) and device.type == "cuda"
    needed = (train_x.numel() + val_x.numel()) * 4
    if cache and needed < torch.cuda.mem_get_info()[0] * 0.25:
        train_x, train_y, train_c = train_x.to(device), train_y.to(device), train_c.to(device)
        val_x = val_x.to(device)
    generator = torch.Generator().manual_seed(seed)
    write_json(output / "config.json", config)
    write_json(output / "normalization.json", data.normalization())
    write_json(
        output / "provenance.json",
        {
            "manifest_sha256": data.manifest_hash,
            "train_ids_sha256": hashlib.sha256(
                train_meta.global_index.to_numpy().tobytes()
            ).hexdigest(),
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "n_train": len(train_x),
            "n_val": len(val_x),
            "parameters": sum(p.numel() for p in model.parameters()),
            "source_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")},
            "training_precision": "bfloat16 autocast" if amp else "float32",
            "inference_precision": "float32",
            "tf32": False,
            "test_evaluated": False,
            "objective": "CE"
            if mode == "classifier"
            else {
                "mse": "standardized marker MSE + lambda * CE",
                "spo_plus": "top-k SPO+ surrogate + lambda * CE",
                "pg_central": "top-k central perturbation-gradient loss + lambda * CE",
            }[objective],
            "decision_training_pool": (
                None
                if objective == "mse"
                else "random minibatch; k=ceil(primary_fraction*batch_cells); normalized by k"
            ),
            "loss_scale": float(config.get("loss_scale", 1.0)),
            "checkpoint_policy": "declared candidate epochs plus fixed final epoch; no early stopping",
            "checkpoint_epochs": candidate_epochs,
            "reference_epoch": reference_epoch,
        },
    )
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    fraction = float(config.get("primary_fraction", 0.1))
    history, candidates = [], []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        loss_sum, mse_sum, ce_sum, seen, active_sum = 0.0, 0.0, 0.0, 0, 0.0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            x = train_x[index].to(device, non_blocking=True)
            y = train_y[index].to(device, non_blocking=True)
            c = train_c[index].to(device, non_blocking=True)
            x = (x - mean) / std
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"
            ):
                out = model(x)
                if out["marker"].shape != y.shape:
                    raise ValueError("Regression shape mismatch: broadcasting forbidden")
                loss, mse, ce = objective_loss(
                    out["marker"], y, out["logits"], c, config, epoch=epoch + 1
                )
                loss = loss * float(config.get("loss_scale", 1.0))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in epoch {epoch + 1}")
            loss.backward()
            if config.get("clip_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["clip_grad_norm"]))
            opt.step()
            n = len(index)
            loss_sum += loss.item() * n
            mse_sum += mse.item() * n
            ce_sum += ce.item() * n
            if objective != "mse":
                active_sum += decision_active_fraction(out["marker"], y, config, epoch + 1) * n
            seen += n
        lr_used = float(opt.param_groups[0]["lr"])
        scheduler.step()
        log = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / seen,
            "train_mse_standardized": mse_sum / seen,
            "train_ce": ce_sum / seen,
            "learning_rate": lr_used,
            "seconds": time.perf_counter() - epoch_start,
        }
        if objective != "mse":
            log["decision_active_fraction"] = active_sum / seen
            log["objective_stage"] = (
                "spo_plus_warmup"
                if objective == "pg_central" and epoch + 1 <= decision_warmup
                else objective
            )
        gradient_every = int(config.get("gradient_every", 20))
        if (
            mode == "joint"
            and lam > 0
            and gradient_every > 0
            and (epoch == 0 or (epoch + 1) % gradient_every == 0)
        ):
            count = min(128, len(train_x))
            diagnostic_x = (train_x[:count].to(device) - mean) / std
            log["shared_gradients"] = shared_gradient_diagnostic(
                model, diagnostic_x, train_y[:count].to(device), train_c[:count].to(device)
            )
        eval_every = int(config.get("eval_every", 5))
        if (
            (epoch + 1) % eval_every == 0
            or epoch == 0
            or epoch + 1 == epochs
            or epoch + 1 in candidate_epochs
        ):
            val_logits, val_marker = predict(model, val_x, mean, std, device, batch_size, amp)
            if mode == "classifier":
                probabilities = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
                scores = probabilities @ data.class_means
            else:
                scores = val_marker.astype(np.float64) * data.target_std + data.target_mean
            labels = val_meta[data.marker].to_numpy()
            cls = val_meta.class_id.to_numpy()
            valid_logits = val_logits if mode == "classifier" or lam > 0 else None
            log["validation"] = prediction_metrics(labels, scores, cls, valid_logits)
            log["validation"]["score_std_in_target_units"] = float(np.std(scores))
            log["validation"].update(
                selection_metrics(labels, scores, cls, fraction, val_meta.global_index.to_numpy())
            )
        if epoch + 1 in candidate_epochs:
            checkpoint_file = f"snapshots/epoch_{epoch + 1:03d}.pt"
            predictions_file = f"snapshots/epoch_{epoch + 1:03d}_validation.npz"
            (output / "snapshots").mkdir(exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "normalization": data.normalization(),
                    "manifest_sha256": data.manifest_hash,
                    "epoch": epoch + 1,
                },
                output / checkpoint_file,
            )
            np.savez_compressed(
                output / predictions_file,
                global_index=val_meta.global_index.to_numpy(),
                score=np.asarray(scores, dtype=np.float64),
                logits=val_logits,
            )
            candidates.append(
                {
                    "epoch": epoch + 1,
                    "checkpoint_file": checkpoint_file,
                    "checkpoint_sha256": file_sha256(output / checkpoint_file),
                    "predictions_file": predictions_file,
                    "predictions_sha256": file_sha256(output / predictions_file),
                    "validation": log["validation"],
                }
            )
        history.append(log)
        with open(output / "history.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(log, allow_nan=False) + "\n")
        print(json.dumps({"run": output.name, **log}), flush=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "normalization": data.normalization(),
            "manifest_sha256": data.manifest_hash,
            "optimizer": opt.state_dict(),
            "epoch": epochs,
        },
        output / "final.pt",
    )
    val_logits, val_marker = predict(model, val_x, mean, std, device, batch_size, amp)
    scores = (
        (torch.softmax(torch.from_numpy(val_logits), dim=1).numpy() @ data.class_means)
        if mode == "classifier"
        else val_marker.astype(np.float64) * data.target_std + data.target_mean
    )
    np.savez_compressed(
        output / "validation_predictions.npz",
        global_index=val_meta.global_index.to_numpy(),
        score=scores,
        logits=val_logits,
    )
    complete = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "epochs": epochs,
        "final_validation": history[-1]["validation"],
        "checkpoint_sha256": file_sha256(output / "final.pt"),
        "validation_predictions_sha256": file_sha256(output / "validation_predictions.npz"),
        "config_sha256": file_sha256(output / "config.json"),
        "normalization_sha256": file_sha256(output / "normalization.json"),
        "candidates": candidates,
        "test_evaluated": False,
    }
    write_json(output / "complete.json", complete)
    return complete
