"""Run definitions and the saved study specification."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_name(config):
    weight = f"_w{config['lambda_binary']:g}" if config["method"] == "joint" else ""
    return f"f{config['fraction']:g}_{config['method']}{weight}_s{config['seed']}"


def make_spec(data, settings):
    updates_per_epoch = math.ceil(len(data.rows("train")) / settings["batch_size"])
    package = Path(__file__).parent
    source = {
        name: file_hash(package / name)
        for name in ("model.py", "training.py", "losses.py", "data.py")
    }
    runs = []
    for fraction in settings["train_fractions"]:
        families = [("mse", 0.0), ("bce", 0.0)] + [
            ("joint", value) for value in settings["joint_weights"]
        ]
        for method, weight in families:
            for seed in settings["seeds"]:
                config = {
                    "method": method,
                    "lambda_binary": weight,
                    "seed": seed,
                    "fraction": fraction,
                    "max_steps": updates_per_epoch * settings["full_data_epochs"],
                    "checkpoint_steps": [
                        updates_per_epoch * e for e in settings["checkpoint_epochs"]
                    ],
                    "warmup_steps": updates_per_epoch * settings["warmup_epochs"],
                    "batch_size": settings["batch_size"],
                    "lr": settings["learning_rate"],
                    "weight_decay": settings["weight_decay"],
                    "min_lr_ratio": settings["min_lr_ratio"],
                    "cpu_threads": settings["cpu_threads"],
                    "source_sha256": source,
                    "data_manifest_sha256": data.manifest_sha256,
                }
                runs.append(config)
    return {
        "settings": settings,
        "data_manifest_sha256": data.manifest_sha256,
        "threshold": data.manifest["threshold"],
        "runs": runs,
    }


def lock_spec(data, settings_path, output):
    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    spec = make_spec(data, settings)
    path = Path(output) / "study.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != spec:
            raise ValueError(
                "Study settings, training source, or data changed; use a new output directory"
            )
    else:
        save_json(path, spec)
    return spec
