"""Fit matched regression heads on frozen CE, MSE, and joint encoders.

This is an exploratory follow-up. Encoder parameters and BatchNorm buffers stay
fixed; training and test evaluation use separate commands.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cellselect.model import build_model, IndependentHead
from cellselect.metrics import prediction_metrics, selection_metrics, paired_seed_field_bootstrap
from cellselect.training import PreparedData, file_sha256, seed_everything, write_json

FAMILIES = {"ce": ("classifier", 0.0), "mse": ("joint", 0.0), "joint1": ("joint", 1.0)}
DEFAULT_CONFIG = {
    "epochs": 200,
    "checkpoint_epochs": [20, 40, 60, 80, 100, 150, 200],
    "batch_size": 256,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "min_lr_ratio": 0.05,
    "feature_dim": 256,
    "dropout_rate": 0.4,
    "readout_seed_offset": 10000,
    "primary_fraction": 0.1,
    "fractions": [0.05, 0.1, 0.2],
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fresh_head(seed, feature_dim=256, dropout_rate=0.4):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return IndependentHead(feature_dim, 1, dropout_rate)


def extract_features(model, images, normalization, device, batch_size=256):
    """Return ordinary NumPy arrays, not autograd-incompatible inference tensors."""
    model.eval()
    model.encoder.requires_grad_(False)
    before = state_digest(model.encoder)
    mean = torch.tensor(normalization["image_mean"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    std = torch.tensor(normalization["image_std"], dtype=torch.float32, device=device)[
        None, :, None, None
    ]
    features = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(images), batch_size):
            x = images[start : start + batch_size].to(device)
            features.append(model.encoder((x - mean) / std).float().cpu().numpy().copy())
    result = np.concatenate(features)
    if before != state_digest(model.encoder):
        raise AssertionError("Frozen encoder parameters or BatchNorm buffers changed")
    if result.shape != (len(images), model.feature_dim) or not np.isfinite(result).all():
        raise ValueError("Invalid frozen features")
    return result, before


@torch.inference_mode()
def head_predictions(head, features, normalization, device, batch_size=256):
    head.eval()
    scores = []
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(features), batch_size):
            x = features[start : start + batch_size].to(device)
            scores.append(head(x).squeeze(-1).float().cpu().numpy())
    return (
        np.concatenate(scores).astype(np.float64) * normalization["target_std"]
        + normalization["target_mean"]
    )


def fit_head(
    train_features,
    target,
    val_features,
    val_target,
    val_classes,
    val_ids,
    normalization,
    config,
    binding,
    output,
    device,
):
    output = Path(output)
    if (output / "complete.json").exists():
        done = _read(output / "complete.json")
        if (
            done["config"] != config
            or done["binding"] != binding
            or done["script_sha256"] != file_sha256(__file__)
        ):
            raise ValueError(
                "Existing readout run has changed configuration, features or implementation"
            )
        for candidate in done["candidates"]:
            for name, checksum in [
                ("checkpoint_file", "checkpoint_sha256"),
                ("predictions_file", "predictions_sha256"),
            ]:
                if file_sha256(output / candidate[name]) != candidate[checksum]:
                    raise ValueError("Readout candidate artifact changed")
        return done
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Partial readout run: {output}")
    epochs, checkpoints, batch_size = (
        config["epochs"],
        config["checkpoint_epochs"],
        config["batch_size"],
    )
    if (
        not isinstance(epochs, int)
        or epochs < 1
        or checkpoints != sorted(set(checkpoints))
        or not checkpoints
        or any(not isinstance(e, int) or not 1 <= e <= epochs for e in checkpoints)
        or batch_size < 1
        or normalization["target_std"] <= 0
    ):
        raise ValueError("Invalid readout schedule or normalization")
    train_features, val_features = (
        np.asarray(train_features, np.float32),
        np.asarray(val_features, np.float32),
    )
    target, val_target = np.asarray(target, np.float32), np.asarray(val_target, float)
    if (
        train_features.shape != (len(target), config["feature_dim"])
        or val_features.shape != (len(val_target), config["feature_dim"])
        or target.ndim != 1
        or val_target.ndim != 1
        or not all(np.isfinite(a).all() for a in [train_features, val_features, target, val_target])
    ):
        raise ValueError("Invalid aligned readout arrays")
    seed_everything(config["readout_seed"])
    head = fresh_head(config["readout_seed"], config["feature_dim"], config["dropout_rate"]).to(
        device
    )
    initial_digest = state_digest(head)
    x, y = (
        torch.from_numpy(train_features.copy()).to(device),
        torch.from_numpy(target.copy()).to(device),
    )
    vx = torch.from_numpy(val_features.copy()).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    warmup, minimum = config["warmup_epochs"], config["min_lr_ratio"]

    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    generator = torch.Generator().manual_seed(config["readout_seed"])
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    candidates, started = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        for index in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = head(x[index]).squeeze(-1)
            if prediction.shape != y[index].shape:
                raise ValueError("Readout regression broadcasting forbidden")
            loss = F.mse_loss(prediction, y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite readout loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        scheduler.step()
        row = {"epoch": epoch, "training_mse": total / len(x)}
        if epoch in checkpoints:
            scores = head_predictions(head, vx, normalization, device, batch_size)
            metrics = prediction_metrics(val_target, scores)
            metrics.update(
                selection_metrics(
                    val_target, scores, val_classes, config["primary_fraction"], val_ids
                )
            )
            checkpoint_file, predictions_file = (
                f"epoch_{epoch:03d}.pt",
                f"epoch_{epoch:03d}_val.npz",
            )
            torch.save(
                {
                    "head": head.state_dict(),
                    "config": config,
                    "normalization": normalization,
                    "binding": binding,
                    "epoch": epoch,
                },
                output / checkpoint_file,
            )
            np.savez_compressed(output / predictions_file, global_index=val_ids, score=scores)
            candidates.append(
                {
                    "epoch": epoch,
                    "validation": metrics,
                    "checkpoint_file": checkpoint_file,
                    "checkpoint_sha256": file_sha256(output / checkpoint_file),
                    "predictions_file": predictions_file,
                    "predictions_sha256": file_sha256(output / predictions_file),
                }
            )
            row["validation"] = metrics
            print(json.dumps({"readout": output.name, **row}), flush=True)
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    done = {
        "config": config,
        "binding": binding,
        "initial_head_sha256": initial_digest,
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "candidates": candidates,
        "script_sha256": file_sha256(__file__),
        "seconds": time.perf_counter() - started,
        "test_used": False,
    }
    write_json(output / "complete.json", done)
    return done


def _sources(core):
    result = []
    for family, (mode, weight) in FAMILIES.items():
        for seed in core["seeds"]:
            matches = [
                r
                for r in core["runs"]
                if r["epoch"] == 40
                and r["config"]["mode"] == mode
                and r["config"]["lambda_class"] == weight
                and r["config"]["seed"] == seed
            ]
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicate fixed-40 source for {family}/{seed}")
            result.append({"family": family, "encoder_seed": seed, "source": matches[0]})
    return result


def _load_encoder(entry, run_root, data, device):
    source = entry["source"]
    directory = Path(run_root) / source["name"]
    for filename, key in [
        (source["checkpoint_file"], "checkpoint_sha256"),
        ("config.json", "config_sha256"),
        ("normalization.json", "normalization_sha256"),
        (source["predictions_file"], "validation_predictions_sha256"),
    ]:
        if file_sha256(directory / filename) != source[key]:
            raise ValueError("Frozen source artifact changed")
    saved = torch.load(directory / source["checkpoint_file"], map_location="cpu", weights_only=True)
    if (
        saved["epoch"] != 40
        or saved["config"] != source["config"]
        or saved["normalization"] != data.normalization()
    ):
        raise ValueError("Encoder epoch/config/normalization mismatch")
    if saved["manifest_sha256"] != data.manifest_hash:
        raise ValueError("Encoder was trained on another cohort")
    model = build_model(feature_dim=256)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval()


def train_and_freeze(data, run_root, core_protocol, output, device, config=None):
    output = Path(output)
    core = _read(core_protocol)
    config = {**DEFAULT_CONFIG, **(config or {})}
    if (
        not core.get("expected_experiment_verified")
        or core.get("test_used_to_choose_protocol") is not False
        or core["manifest_sha256"] != data.manifest_hash
    ):
        raise ValueError("Require the matching verified frozen core protocol")
    if (output / "protocol.json").exists():
        raise FileExistsError("A frozen readout protocol already exists; do not replace it")
    output.mkdir(parents=True, exist_ok=True)
    declared = {
        "core_protocol_sha256": file_sha256(core_protocol),
        "config": config,
        "encoder_epoch": 40,
        "families": FAMILIES,
        "encoder_seeds": core["seeds"],
        "existing_main_test_previously_evaluated": True,
        "status": "exploratory matched readout",
    }
    # JSON canonicalization converts tuple-valued family definitions to lists.
    declared = json.loads(json.dumps(declared))
    declaration_file = output / "declaration.json"
    if declaration_file.exists() and _read(declaration_file) != declared:
        raise ValueError("Existing probe declaration differs")
    write_json(declaration_file, declared)
    train_x, train_y, _, train_meta = data.partition("train")
    val_x, _, _, val_meta = data.partition("val")
    runs = []
    for entry in _sources(core):
        family, seed = entry["family"], entry["encoder_seed"]
        name = f"{family}_seed_{seed}"
        cache_file, cache_meta = output / f"{name}_features.npz", output / f"{name}_features.json"
        binding = {
            "manifest_sha256": data.manifest_hash,
            "source_checkpoint_sha256": entry["source"]["checkpoint_sha256"],
            "normalization": data.normalization(),
            "script_sha256": file_sha256(__file__),
        }
        if cache_meta.exists():
            cached = _read(cache_meta)
            if cached["binding"] != binding or file_sha256(cache_file) != cached["features_sha256"]:
                raise ValueError("Frozen feature cache changed")
            model = _load_encoder(entry, run_root, data, device)
            if state_digest(model.encoder) != cached["encoder_state_sha256"]:
                raise ValueError("Cached source encoder changed")
            del model
        else:
            model = _load_encoder(entry, run_root, data, device)
            train_features, digest = extract_features(
                model, train_x, data.normalization(), device, config["batch_size"]
            )
            val_features, after = extract_features(
                model, val_x, data.normalization(), device, config["batch_size"]
            )
            if digest != after:
                raise AssertionError("Encoder state changed between partitions")
            np.savez_compressed(
                cache_file,
                train=train_features,
                val=val_features,
                train_ids=train_meta.global_index.to_numpy(),
                val_ids=val_meta.global_index.to_numpy(),
            )
            cached = {
                "binding": binding,
                "encoder_state_sha256": digest,
                "features_sha256": file_sha256(cache_file),
            }
            write_json(cache_meta, cached)
            del model
        with np.load(cache_file, allow_pickle=False) as features:
            np.testing.assert_array_equal(features["train_ids"], train_meta.global_index.to_numpy())
            np.testing.assert_array_equal(features["val_ids"], val_meta.global_index.to_numpy())
            run_config = {
                **config,
                "family": family,
                "encoder_seed": seed,
                "readout_seed": config["readout_seed_offset"] + seed,
            }
            run_binding = {
                **binding,
                "features_sha256": cached["features_sha256"],
                "encoder_state_sha256": cached["encoder_state_sha256"],
            }
            done = fit_head(
                features["train"],
                train_y.numpy(),
                features["val"],
                val_meta[data.marker].to_numpy(),
                val_meta.class_id.to_numpy(),
                val_meta.global_index.to_numpy(),
                data.normalization(),
                run_config,
                run_binding,
                output / name,
                device,
            )
        runs.append(
            {
                **entry,
                "name": name,
                "complete_sha256": file_sha256(output / name / "complete.json"),
                "initial_head_sha256": done["initial_head_sha256"],
                "binding": run_binding,
                "candidates": done["candidates"],
            }
        )
    for seed in core["seeds"]:
        if len({r["initial_head_sha256"] for r in runs if r["encoder_seed"] == seed}) != 1:
            raise AssertionError("Paired families did not start with identical fresh heads")
    choices = {}
    for family in FAMILIES:
        family_runs = [r for r in runs if r["family"] == family]
        by_epoch = {
            epoch: float(
                np.mean(
                    [
                        next(
                            c["validation"]["utility"]
                            for c in r["candidates"]
                            if c["epoch"] == epoch
                        )
                        for r in family_runs
                    ]
                )
            )
            for epoch in config["checkpoint_epochs"]
        }
        choices[family] = max(by_epoch, key=lambda e: (by_epoch[e], -e))
    protocol = {
        **declared,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": data.manifest_hash,
        "marker": data.marker,
        "runs": runs,
        "selected_head_epochs": choices,
        "selection": "One epoch per encoder family maximizes mean validation utility across paired seeds; ties use earlier epoch",
        "test_used_for_readout_selection": False,
        "interpretation": "Fresh identical nonlinear readouts on frozen representations. Rescue supports a score/readout restriction; non-rescue does not prove irreversible information loss.",
    }
    write_json(output / "protocol.json", protocol)
    return protocol


def evaluate_probe(data, run_root, output, device, n_bootstrap=2000):
    """Only this explicit stage opens test images or outcomes."""
    output = Path(output)
    protocol_file = output / "protocol.json"
    protocol = _read(protocol_file)
    if (
        protocol["manifest_sha256"] != data.manifest_hash
        or protocol["test_used_for_readout_selection"] is not False
    ):
        raise ValueError("Require a matching validation-only readout freeze")
    destination = output / "test_results.json"
    if destination.exists():
        raise FileExistsError("Do not replace completed readout evaluation")
    # Validate every selected readout before opening the test partition.
    selected = []
    for run in protocol["runs"]:
        directory = output / run["name"]
        if file_sha256(directory / "complete.json") != run["complete_sha256"]:
            raise ValueError("Readout completion metadata changed")
        epoch = protocol["selected_head_epochs"][run["family"]]
        candidate = next(c for c in run["candidates"] if c["epoch"] == epoch)
        for filename, checksum in [
            (candidate["checkpoint_file"], candidate["checkpoint_sha256"]),
            (candidate["predictions_file"], candidate["predictions_sha256"]),
        ]:
            if file_sha256(directory / filename) != checksum:
                raise ValueError("Selected readout artifact changed")
        model = _load_encoder(run, run_root, data, device)
        if state_digest(model.encoder) != run["binding"]["encoder_state_sha256"]:
            raise ValueError("Frozen encoder state changed")
        del model
        selected.append((run, candidate))
    write_json(
        output / "test_started.json",
        {
            "protocol_sha256": file_sha256(protocol_file),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    x, _, _, metadata = data.partition("test")
    y, classes, ids = (
        metadata[data.marker].to_numpy(),
        metadata.class_id.to_numpy(),
        metadata.global_index.to_numpy(),
    )
    score_map, rows = {}, []
    for run, candidate in selected:
        model = _load_encoder(run, run_root, data, device)
        features, digest = extract_features(
            model, x, data.normalization(), device, protocol["config"]["batch_size"]
        )
        if digest != run["binding"]["encoder_state_sha256"]:
            raise AssertionError("Encoder changed during test feature extraction")
        tensor = torch.from_numpy(features.copy()).to(device)
        saved = torch.load(
            output / run["name"] / candidate["checkpoint_file"],
            map_location="cpu",
            weights_only=True,
        )
        if saved["binding"] != run["binding"] or saved["epoch"] != candidate["epoch"]:
            raise ValueError("Selected head checkpoint binding mismatch")
        head = fresh_head(saved["config"]["readout_seed"], 256, saved["config"]["dropout_rate"]).to(
            device
        )
        head.load_state_dict(saved["head"], strict=True)
        fresh = head_predictions(
            head, tensor, saved["normalization"], device, saved["config"]["batch_size"]
        )
        if run["family"] == "ce":
            with torch.inference_mode():
                logits = np.concatenate(
                    [
                        model.classification_head(part).float().cpu().numpy()
                        for part in tensor.split(saved["config"]["batch_size"])
                    ]
                )
            # Match the original pipeline's CPU softmax rather than introducing
            # device-dependent rounding into its native baseline.
            native = torch.softmax(torch.from_numpy(logits), dim=1).numpy() @ data.class_means
        else:
            native = head_predictions(
                model.regression_head,
                tensor,
                data.normalization(),
                device,
                saved["config"]["batch_size"],
            )
        np.savez_compressed(
            output / f"{run['name']}_test_predictions.npz",
            global_index=ids,
            fresh_readout=fresh,
            native_score=native,
        )
        for kind, score in [("fresh_readout", fresh), ("native_score", native)]:
            score_map[(run["family"], kind, run["encoder_seed"])] = score
            metrics = prediction_metrics(y, score)
            for fraction in protocol["config"]["fractions"]:
                rows.append(
                    {
                        "family": run["family"],
                        "kind": kind,
                        "encoder_seed": run["encoder_seed"],
                        "readout_seed": saved["config"]["readout_seed"],
                        "head_epoch": saved["epoch"] if kind == "fresh_readout" else None,
                        "encoder_epoch": 40,
                        **metrics,
                        **selection_metrics(y, score, classes, fraction, ids),
                    }
                )
        del model, head, tensor
    contrasts = [
        ("ce_readout_minus_native_ce", ("ce", "fresh_readout"), ("ce", "native_score")),
        ("ce_readout_minus_mse_readout", ("ce", "fresh_readout"), ("mse", "fresh_readout")),
        ("joint_readout_minus_mse_readout", ("joint1", "fresh_readout"), ("mse", "fresh_readout")),
        ("mse_readout_minus_native_mse", ("mse", "fresh_readout"), ("mse", "native_score")),
        (
            "joint_readout_minus_native_joint",
            ("joint1", "fresh_readout"),
            ("joint1", "native_score"),
        ),
    ]
    comparisons = []
    for name, a, b in contrasts:
        first = np.stack([score_map[(*a, seed)] for seed in protocol["encoder_seeds"]])
        second = np.stack([score_map[(*b, seed)] for seed in protocol["encoder_seeds"]])
        interval = paired_seed_field_bootstrap(
            y,
            first,
            second,
            metadata.fov_id.to_numpy(),
            protocol["config"]["primary_fraction"],
            n_bootstrap=n_bootstrap,
            ids=ids,
        )
        interval["bootstrap_seed"] = interval.pop("seed")
        comparisons.append({"comparison": name, **interval})
    result = {
        "protocol_sha256": file_sha256(protocol_file),
        "existing_main_test_previously_evaluated": True,
        "status": "exploratory matched frozen-readout test evaluation",
        "n_test": len(y),
        "n_fields": int(metadata.fov_id.nunique()),
        "encoder_seeds": protocol["encoder_seeds"],
        "selected_head_epochs": protocol["selected_head_epochs"],
        "metrics": rows,
        "comparisons": comparisons,
        "limits": "Matched nonlinear probes test accessibility under this head and training procedure, not total information content. Field intervals condition on frozen encoders/readouts and selected epochs; three paired seeds do not separate encoder-seed from readout-seed variability.",
    }
    write_json(destination, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "test"])
    for name in ["data", "runs", "core-protocol", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    data = PreparedData(args.data)
    if args.stage == "train":
        result = train_and_freeze(data, args.runs, args.core_protocol, args.output, device)
        print(
            "Readout choices frozen; no test data used:", result["selected_head_epochs"], flush=True
        )
    else:
        if _read(args.output / "protocol.json")["core_protocol_sha256"] != file_sha256(
            args.core_protocol
        ):
            raise ValueError("Core protocol differs from the readout declaration")
        result = evaluate_probe(data, args.runs, args.output, device, args.bootstrap)
        print("Readout evaluation completed:", len(result["metrics"]), "metric rows", flush=True)


if __name__ == "__main__":
    main()
