"""Continue the CD45 classifier under a fixed validation-only protocol.

Restore epoch-200 weights and AdamW moments, then use a declared fresh RNG
stream and fixed terminal learning rate. This is not an uninterrupted replay.
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
from cellselect.model import build_model
from cellselect.metrics import selection_metrics
from cellselect.study import class_mean_scores, study_prediction_metrics
from cellselect.training import PreparedData, file_sha256, predict, seed_everything, write_json

SETTINGS = {
    "start_epoch": 200,
    "end_epoch": 400,
    "candidate_epochs": [200, 250, 300, 350, 400],
    "seeds": [16, 26, 36],
    "rng_seed_offset": 200000,
    "learning_rate": 1.5e-5,
    "batch_size": 256,
    "primary_fraction": 0.1,
    "marker": "CD45",
    "train_fraction": 1.0,
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sources():
    files = [Path(__file__)] + [
        ROOT / "src/cellselect" / name
        for name in ("model.py", "training.py", "metrics.py", "study.py", "data.py")
    ]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in files}


def restore_source(run_root, record, learning_rate, device):
    """Restore every AdamW moment/step; replace only its post-scheduler LR."""
    directory = Path(run_root) / record["name"]
    for name, digest in record["files_sha256"].items():
        require(file_sha256(directory / name) == digest, f"Bound source artifact changed: {name}")
    saved = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)
    require(
        saved["epoch"] == 200 and saved["config"] == record["config"] and "optimizer" in saved,
        "Source epoch/config/optimizer mismatch",
    )
    config = saved["config"]
    model = build_model(seed=config["seed"], feature_dim=config["feature_dim"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    optimizer.load_state_dict(saved["optimizer"])
    require(bool(optimizer.state), "Source optimizer has no restored moments")
    saved_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return model, optimizer, saved, saved_lrs


def optimizer_steps(optimizer):
    steps = [int(state["step"].item()) for state in optimizer.state.values()]
    require(
        bool(steps) and len(set(steps)) == 1, "Optimizer parameters have inconsistent update counts"
    )
    return steps[0]


def declare(data_root, run_root, protocol_file, output):
    output, run_root = Path(output), Path(run_root)
    require(
        not output.resolve().is_relative_to(run_root.resolve()),
        "Continuation output cannot modify the main run tree",
    )
    require(
        not output.exists() or not any(output.iterdir()), "Use a new empty continuation directory"
    )
    main = read(protocol_file)
    records = []
    for seed in SETTINGS["seeds"]:
        name = f"CD45_f1_seed_{seed}_class_means"
        directory = run_root / name
        config, complete, provenance = [
            read(directory / x) for x in ("config.json", "complete.json", "provenance.json")
        ]
        require(
            config["marker"] == "CD45"
            and config["train_fraction"] == 1
            and config["seed"] == seed
            and config["mode"] == "classifier"
            and config["epochs"] == 200
            and config["batch_size"] == 256
            and config["amp"] is False
            and config.get("loss_scale", 1) == 1
            and config.get("clip_grad_norm") is None,
            "Source is not the declared full-cohort FP32 classifier",
        )
        frozen = [r for r in main["records"] if r["name"] == name and r["epoch"] == 200]
        require(
            len(frozen) == 1 and frozen[0]["config"] == config,
            "Source is absent from the frozen main study",
        )
        for filename, key in [
            ("final.pt", "checkpoint_sha256"),
            ("validation_predictions.npz", "validation_predictions_sha256"),
            ("config.json", "config_sha256"),
            ("normalization.json", "normalization_sha256"),
        ]:
            require(
                file_sha256(directory / filename) == complete[key],
                "Main completed artifact changed",
            )
        history = [
            json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()
        ]
        require(
            [r["epoch"] for r in history] == list(range(1, 201))
            and np.isclose(
                history[-1]["learning_rate"], SETTINGS["learning_rate"], rtol=0, atol=1e-18
            ),
            "Source terminal USED learning rate differs",
        )
        require(
            np.isclose(
                config["learning_rate"] * config["min_lr_ratio"],
                SETTINGS["learning_rate"],
                rtol=0,
                atol=1e-18,
            ),
            "Source schedule minimum differs",
        )
        for filename in ("model.py", "training.py", "metrics.py", "data.py"):
            require(
                provenance["source_sha256"][filename]
                == file_sha256(ROOT / "src/cellselect" / filename),
                "Source code changed",
            )
        data = PreparedData(
            data_root, marker="CD45", train_fraction=1, subset_seed=config["subset_seed"]
        )
        train = data.metadata.iloc[data.indices["train"]]
        require(
            provenance["manifest_sha256"] == data.manifest_hash == main["manifest_sha256"]
            and read(directory / "normalization.json") == data.normalization()
            and provenance["train_ids_sha256"]
            == hashlib.sha256(train.global_index.to_numpy().tobytes()).hexdigest(),
            "Training cohort or normalization changed",
        )
        files = [
            "final.pt",
            "validation_predictions.npz",
            "config.json",
            "normalization.json",
            "complete.json",
            "provenance.json",
            "history.jsonl",
        ]
        record = {
            "name": name,
            "seed": seed,
            "config": config,
            "rng_seed": seed + SETTINGS["rng_seed_offset"],
            "files_sha256": {name: file_sha256(directory / name) for name in files},
            "terminal_used_lr": history[-1]["learning_rate"],
            "n_train": len(train),
            "n_val": len(data.indices["val"]),
        }
        model, opt, saved, lrs = restore_source(
            run_root, record, SETTINGS["learning_rate"], torch.device("cpu")
        )
        require(
            saved["normalization"] == data.normalization()
            and saved["manifest_sha256"] == data.manifest_hash,
            "Restored checkpoint normalization differs",
        )
        record.update(saved_optimizer_lr=lrs, initial_optimizer_updates=optimizer_steps(opt))
        require(
            record["initial_optimizer_updates"] == 200 * math.ceil(len(train) / 256),
            "Source optimizer update count differs",
        )
        records.append(record)
        del model, opt, saved
    declaration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS,
        "runs": records,
        "main_protocol_sha256": file_sha256(protocol_file),
        "manifest_sha256": data.manifest_hash,
        "source_sha256": sources(),
        "selection": "One common epoch maximizing mean validation top-10% utility across three paired seeds; ties choose the earliest epoch.",
        "scope": "Secondary validation-only continuation robustness check; main108 unchanged, no test or external evaluation. No extension after epoch400.",
        "rng": "Fresh declared seed=original seed+200000; original RNG state was not stored, so this is not an exact uninterrupted trajectory.",
        "schedule": "AdamW moments and update counts restored; fixed last USED learning rate, with post-epoch200 scheduler LR explicitly replaced.",
    }
    write_json(output / "declaration.json", declaration)
    return declaration


def continue_run(data, run_root, record, declaration, output, device):
    output = Path(output)
    require(
        not output.exists() or not any(output.iterdir()),
        "Partial/completed continuation exists; do not silently restart",
    )
    require(
        declaration["settings"] == SETTINGS and declaration["source_sha256"] == sources(),
        "Continuation declaration/code changed",
    )
    model, opt, saved, lrs = restore_source(run_root, record, SETTINGS["learning_rate"], device)
    require(
        saved["normalization"] == data.normalization()
        and saved["manifest_sha256"] == data.manifest_hash
        and data.manifest_hash == declaration["manifest_sha256"],
        "Continuation data or normalization differs",
    )
    initial_updates = optimizer_steps(opt)
    require(initial_updates == record["initial_optimizer_updates"], "Restored update count changed")
    seed_everything(record["rng_seed"])
    generator = torch.Generator().manual_seed(record["rng_seed"])
    tx, _, tc, tm = data.partition("train")
    vx, _, _, vm = data.partition("val")
    require(
        len(tx) == record["n_train"] and len(vx) == record["n_val"],
        "Train/validation cohort size changed",
    )
    if (
        device.type == "cuda"
        and (tx.numel() + vx.numel()) * 4 < torch.cuda.mem_get_info()[0] * 0.25
    ):
        tx, tc, vx = tx.to(device), tc.to(device), vx.to(device)
    mean = torch.tensor(data.image_mean, dtype=torch.float32, device=device)[None, :, None, None]
    std = torch.tensor(data.image_std, dtype=torch.float32, device=device)[None, :, None, None]
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "provenance.json",
        {
            "source": record,
            "source_sha256": sources(),
            "settings": SETTINGS,
            "manifest_sha256": data.manifest_hash,
            "train_ids_sha256": hashlib.sha256(tm.global_index.to_numpy().tobytes()).hexdigest(),
            "restored_optimizer_lrs": lrs,
            "fixed_lr": SETTINGS["learning_rate"],
            "test_used": False,
            "precision": "float32, TF32 disabled",
            "device": str(device),
            "torch_version": torch.__version__,
        },
    )
    candidates = []
    for epoch in range(200, 401):
        started = time.perf_counter()
        loss_sum = 0.0
        if epoch > 200:
            model.train()
            for index in torch.randperm(len(tx), generator=generator).split(256):
                opt.zero_grad(set_to_none=True)
                logits = model((tx[index].to(device) - mean) / std)["logits"]
                loss = F.cross_entropy(logits, tc[index].to(device))
                require(bool(torch.isfinite(loss)), "Nonfinite CE continuation loss")
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach()) * len(index)
            require(
                all(group["lr"] == SETTINGS["learning_rate"] for group in opt.param_groups),
                "Fixed LR changed",
            )
        updates = optimizer_steps(opt)
        require(
            updates == initial_updates + (epoch - 200) * math.ceil(len(tx) / 256),
            "Optimizer update continuity failed",
        )
        row = {
            "epoch": epoch,
            "learning_rate": SETTINGS["learning_rate"],
            "optimizer_updates": updates,
            "train_ce": loss_sum / len(tx) if epoch > 200 else None,
            "seconds": time.perf_counter() - started,
        }
        if epoch in SETTINGS["candidate_epochs"]:
            if epoch == 200:
                with np.load(
                    Path(run_root) / record["name"] / "validation_predictions.npz",
                    allow_pickle=False,
                ) as values:
                    require(
                        np.array_equal(values["global_index"], vm.global_index.to_numpy()),
                        "Source validation IDs differ",
                    )
                    logits = values["logits"].copy()
            else:
                logits, _ = predict(model, vx, mean, std, device, 256, amp=False)
            score = class_mean_scores(logits, data.class_means)
            metrics = study_prediction_metrics(
                vm.CD45.to_numpy(), score, vm.class_id.to_numpy(), logits
            )
            metrics.update(
                selection_metrics(
                    vm.CD45.to_numpy(),
                    score,
                    vm.class_id.to_numpy(),
                    0.1,
                    vm.global_index.to_numpy(),
                )
            )
            pred = f"epoch_{epoch:03d}_validation.npz"
            np.savez_compressed(
                output / pred, global_index=vm.global_index.to_numpy(), score=score, logits=logits
            )
            state = f"epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "normalization": data.normalization(),
                    "manifest_sha256": data.manifest_hash,
                    "source": record,
                    "settings": SETTINGS,
                    "epoch": epoch,
                },
                output / state,
            )
            candidates.append(
                {
                    "epoch": epoch,
                    "validation": metrics,
                    "predictions_file": pred,
                    "predictions_sha256": file_sha256(output / pred),
                    "checkpoint_file": state,
                    "checkpoint_sha256": file_sha256(output / state),
                    "optimizer_updates": updates,
                }
            )
            row["validation"] = metrics
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if epoch in SETTINGS["candidate_epochs"] or epoch % 10 == 0:
            print(json.dumps({"run": record["name"], **row}), flush=True)
    complete = {
        "seed": record["seed"],
        "candidates": candidates,
        "source_sha256": sources(),
        "test_used": False,
        "source_checkpoint_sha256": record["files_sha256"]["final.pt"],
        "total_added_optimizer_updates": updates - initial_updates,
    }
    write_json(output / "complete.json", complete)
    return complete


def train(data_root, run_root, protocol_file, output, device, cpu_threads=8):
    output = Path(output)
    declaration = read(output / "declaration.json")
    require(
        declaration["main_protocol_sha256"] == file_sha256(protocol_file),
        "Frozen main protocol changed",
    )
    require(
        declaration["settings"] == SETTINGS and declaration["source_sha256"] == sources(),
        "Declared code/settings changed",
    )
    require(not (output / "protocol.json").exists(), "Continuation choices are already frozen")
    torch.set_num_threads(cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    done = []
    for record in declaration["runs"]:
        data = PreparedData(data_root, "CD45", 1.0, record["config"]["subset_seed"])
        done.append(
            continue_run(
                data, run_root, record, declaration, output / record["name"], torch.device(device)
            )
        )
    rows = [
        {
            "epoch": epoch,
            "validation_utility_by_seed": [
                next(c for c in r["candidates"] if c["epoch"] == epoch)["validation"]["utility"]
                for r in done
            ],
            "validation_rmse_by_seed": [
                next(c for c in r["candidates"] if c["epoch"] == epoch)["validation"]["rmse"]
                for r in done
            ],
        }
        for epoch in SETTINGS["candidate_epochs"]
    ]
    for row in rows:
        row["mean_validation_utility"] = float(np.mean(row["validation_utility_by_seed"]))
        row["mean_validation_rmse"] = float(np.mean(row["validation_rmse_by_seed"]))
    best = max(rows, key=lambda row: row["mean_validation_utility"])
    result = {
        "declaration_sha256": file_sha256(output / "declaration.json"),
        "seeds": SETTINGS["seeds"],
        "candidates": rows,
        "selected_epoch": best["epoch"],
        "selection": declaration["selection"],
        "test_used": False,
        "completed_runs": [
            {
                "name": record["name"],
                "complete_sha256": file_sha256(output / record["name"] / "complete.json"),
            }
            for record in declaration["runs"]
        ],
        "scope": declaration["scope"],
    }
    write_json(output / "protocol.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["declare", "train"])
    parser.add_argument("--data", type=Path, default=Path("data/prepared/bsccmnist_cd16_v1"))
    parser.add_argument("--runs", type=Path, default=Path("results/runs/extended"))
    parser.add_argument("--protocol", type=Path, default=Path("results/extended_protocol.json"))
    parser.add_argument("--output", type=Path, default=Path("results/cd45_classifier_continuation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "declare":
        result = declare(args.data, args.runs, args.protocol, args.output)
        print(
            json.dumps(
                {
                    "declared_seeds": SETTINGS["seeds"],
                    "candidate_epochs": SETTINGS["candidate_epochs"],
                    "training_started": False,
                }
            )
        )
    else:
        result = train(
            args.data, args.runs, args.protocol, args.output, args.device, args.cpu_threads
        )
        print(json.dumps({"selected_epoch": result["selected_epoch"], "test_used": False}))


if __name__ == "__main__":
    main()
