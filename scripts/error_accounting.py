"""Post-hoc accounting of predictive-loss versus utility-selected policies.

Loss gains are old minus new; selection changes are new minus old. Each group's
loss contribution uses the full candidate-pool size as its denominator.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked(path, expected):
    path = Path(path)
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    require(actual == expected, f"Changed input: {path.name}")
    return path


def account(target, old_logits, new_logits, ids, threshold, fraction=0.1):
    target, old_logits, new_logits = [
        np.asarray(v, dtype=float) for v in (target, old_logits, new_logits)
    ]
    ids = np.asarray(ids)
    require(
        target.ndim == 1
        and len(target)
        and all(v.shape == target.shape for v in (old_logits, new_logits, ids)),
        "Misaligned arrays",
    )
    require(
        all(np.isfinite(v).all() for v in (target, old_logits, new_logits))
        and np.isfinite(threshold)
        and 0 < fraction <= 1
        and np.issubdtype(ids.dtype, np.integer)
        and len(np.unique(ids)) == len(ids),
        "Invalid scores, IDs or budget",
    )
    n, k = len(target), int(np.ceil(fraction * len(target)))
    high = target >= threshold
    masks = []
    for scores in (old_logits, new_logits):
        chosen = sorted(range(n), key=lambda i: (-float(scores[i]), int(ids[i])))[:k]
        mask = np.zeros(n, dtype=bool)
        mask[chosen] = True
        masks.append(mask)
    old, new = masks
    groups = {
        "selected_by_both": old & new,
        "exchanged": old ^ new,
        "selected_by_neither": ~(old | new),
    }
    require(
        np.all(sum(mask.astype(int) for mask in groups.values()) == 1),
        "Groups do not partition the pool",
    )
    loss = lambda z: np.where(high, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))
    brier = lambda z: (np.exp(-np.logaddexp(0.0, -z)) - high) ** 2
    old_loss, new_loss, old_brier, new_brier = (
        loss(old_logits),
        loss(new_logits),
        brier(old_logits),
        brier(new_logits),
    )
    log_gain, brier_gain = old_loss - new_loss, old_brier - new_brier
    rows = [
        {
            "group": name,
            "n_cells": n,
            "group_cells": int(mask.sum()),
            "group_fraction": float(mask.mean()),
            "log_loss_gain_per_pool_cell": float(log_gain[mask].sum() / n),
            "brier_gain_per_pool_cell": float(brier_gain[mask].sum() / n),
        }
        for name, mask in groups.items()
    ]
    require(
        np.isclose(
            sum(r["log_loss_gain_per_pool_cell"] for r in rows), log_gain.mean(), rtol=0, atol=1e-12
        )
        and np.isclose(
            sum(r["brier_gain_per_pool_cell"] for r in rows), brier_gain.mean(), rtol=0, atol=1e-12
        ),
        "Loss contributions do not sum",
    )
    added, removed = new & ~old, old & ~new
    mean_change = float(target[new].mean() - target[old].mean())
    purity_change = float(high[new].mean() - high[old].mean())
    marker_exchange = float(target[added].sum() - target[removed].sum())
    hit_exchange = int(high[added].sum()) - int(high[removed].sum())
    require(
        added.sum() == removed.sum()
        and np.isclose(mean_change, marker_exchange / k, rtol=0, atol=1e-10)
        and np.isclose(purity_change, hit_exchange / k, rtol=0, atol=1e-12),
        "Selection exchange identity failed",
    )
    totals = {
        "n_cells": n,
        "n_selected": k,
        "old_high_selected": int(high[old].sum()),
        "new_high_selected": int(high[new].sum()),
        "exchanged_per_policy": int(added.sum()),
        "mean_before": float(target[old].mean()),
        "mean_after": float(target[new].mean()),
        "mean_change": mean_change,
        "purity_before": float(high[old].mean()),
        "purity_after": float(high[new].mean()),
        "purity_change": purity_change,
        "exchanged_marker_sum_change": marker_exchange,
        "exchanged_high_count_change": hit_exchange,
        "log_loss_before": float(old_loss.mean()),
        "log_loss_after": float(new_loss.mean()),
        "log_loss_gain": float(log_gain.mean()),
        "brier_before": float(old_brier.mean()),
        "brier_after": float(new_brier.mean()),
        "brier_gain": float(brier_gain.mean()),
    }
    return rows, totals


def run(study, output):
    study, output = Path(study), Path(output)
    roots = [study, study / "stopping"]
    selections = [read(root / "selection.json") for root in roots]
    checked(study / "selection.json", selections[1]["primary_selection_sha256"])
    threshold = selections[0]["threshold"]
    require(
        threshold == selections[1]["threshold"]
        and selections[0]["data_manifest_sha256"] == selections[1]["data_manifest_sha256"],
        "Incompatible policy tasks",
    )
    summaries = []
    for pool in ("test", "external"):
        frames, maps = [], []
        for root, selection in zip(roots, selections):
            evaluation = read(root / pool / "evaluation.json")
            require(evaluation["status"] == "complete", "Evaluation is incomplete")
            checked(root / "selection.json", evaluation["selection_sha256"])
            require(
                evaluation["data_manifest_sha256"] == selection["data_manifest_sha256"],
                "Evaluation data differ",
            )
            frames.append(
                pd.read_csv(
                    checked(root / pool / "cells.csv", evaluation["cells_sha256"]),
                    float_precision="round_trip",
                )
            )
            entries = {r["checkpoint"]: r for r in evaluation["prediction_files"]}
            require(
                len(entries) == len(evaluation["prediction_files"]), "Duplicate prediction binding"
            )
            maps.append(entries)
        pd.testing.assert_frame_equal(frames[0], frames[1])
        ids, target = frames[0].global_index.to_numpy(), frames[0].target.to_numpy(dtype=float)
        group_rows, exchange_rows = [], []
        for fraction in (0.25, 1.0):
            for method in ("bce", "joint"):
                choices = []
                for selection in selections:
                    matched = [
                        c
                        for c in selection["selections"]
                        if c["train_fraction"] == fraction
                        and c["method"] == method
                        and c["selected_for"] == "purity"
                    ]
                    require(
                        len(matched) == 1 and matched[0]["score"] == "hit_logit",
                        "Missing probability-score policy",
                    )
                    choices.append(matched[0])
                seeds = [s["seed"] for s in choices[0]["states"]]
                require(
                    seeds == [16, 26, 36] and seeds == [s["seed"] for s in choices[1]["states"]],
                    "Unpaired training seeds",
                )
                for j, seed in enumerate(seeds):
                    logits = []
                    for root, choice, mapping in zip(roots, choices, maps):
                        entry = mapping[choice["states"][j]["checkpoint"]]
                        with np.load(
                            checked(root / pool / entry["file"], entry["sha256"]),
                            allow_pickle=False,
                        ) as saved:
                            require(
                                np.array_equal(saved["global_index"], ids), "Prediction IDs differ"
                            )
                            logits.append(saved["hit_logit"].copy())
                    groups, totals = account(target, *logits, ids, threshold)
                    context = {
                        "analysis": "post_hoc",
                        "train_fraction": fraction,
                        "method": method,
                        "selected_for": "purity",
                        "seed": seed,
                        "selection_fraction": 0.1,
                        "old_step": choices[0]["step"],
                        "new_step": choices[1]["step"],
                        "old_weight": choices[0]["weight"],
                        "new_weight": choices[1]["weight"],
                    }
                    group_rows.extend(context | row for row in groups)
                    exchange_rows.append(context | totals)
        destination = output / ("external/tables" if pool == "external" else "tables")
        destination.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(group_rows).to_csv(destination / "error_accounting.csv", index=False)
        pd.DataFrame(exchange_rows).to_csv(destination / "error_exchanges.csv", index=False)
        quarter = pd.DataFrame(exchange_rows).query("train_fraction == .25 and method == 'bce'")
        summaries.append(
            {
                "pool": pool,
                "group_rows": len(group_rows),
                "exchange_rows": len(exchange_rows),
                **quarter[["log_loss_gain", "brier_gain", "mean_change", "purity_change"]]
                .mean()
                .to_dict(),
            }
        )
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/study"))
    parser.add_argument("--output", type=Path, default=Path("results/stopping"))
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.output), indent=2, allow_nan=False))
