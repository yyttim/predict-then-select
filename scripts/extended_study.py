"""Freeze study choices, evaluate full candidate pools, and compute uncertainty."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cellselect.bootstrap import paired_topk_bootstrap
from cellselect.study import freeze_study, evaluate_study, read_json
from cellselect.training import file_sha256, write_json


def exchange_diagnostics(y, scores_a, scores_b, ids, fraction):
    """Observed utility differences arise only from cells exchanged by the policies.

    Counts are descriptive, not effective sample sizes or estimates of Bayes
    margins. Shared cells cancel even when their marker measurements are noisy.
    """
    from cellselect.metrics import selection_size, topk_indices

    y, ids = np.asarray(y), np.asarray(ids)
    k = selection_size(len(y), fraction)
    records = []
    for a, b in zip(scores_a, scores_b):
        selected_a, selected_b = topk_indices(a, k, ids), topk_indices(b, k, ids)
        added = np.setdiff1d(selected_a, selected_b)
        removed = np.setdiff1d(selected_b, selected_a)
        if len(added) != len(removed):
            raise AssertionError("Compared policies must use the same budget")
        delta = float(y[selected_a].mean() - y[selected_b].mean())
        recovered = float((y[added].sum() - y[removed].sum()) / k)
        if not np.isclose(delta, recovered, atol=1e-8, rtol=1e-10):
            raise AssertionError("Utility exchange identity failed")
        records.append(
            {
                "swapped_cells_per_policy": len(added),
                "selected_overlap_fraction": 1 - len(added) / k,
                "mean_added_outcome": float(y[added].mean()) if len(added) else None,
                "mean_removed_outcome": float(y[removed].mean()) if len(removed) else None,
                "utility_difference": delta,
                "difference_from_exchanges": recovered,
            }
        )
    return {
        "k": k,
        "paired_replicates": records,
        "mean_selected_overlap_fraction": float(
            np.mean([r["selected_overlap_fraction"] for r in records])
        ),
    }


def uncertainty(data_root, protocol_file, result_root, external_root=None, repetitions=2000):
    root = Path(result_root)
    protocol, results = read_json(protocol_file), read_json(root / "results.json")
    if file_sha256(protocol_file) != results["protocol_sha256"]:
        raise ValueError("Evaluation did not use this protocol")
    if file_sha256(Path(data_root) / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("Main manifest changed")
    manifest = read_json(Path(data_root) / "manifest.json")
    if file_sha256(Path(data_root) / "metadata.csv") != manifest["prepared_sha256"]["metadata.csv"]:
        raise ValueError("Main evaluation metadata changed")
    target = root / "uncertainty.json"
    if target.exists():
        raise FileExistsError("Refusing to overwrite uncertainty results")
    metadata = pd.read_csv(Path(data_root) / "metadata.csv")
    test = metadata.loc[metadata.split == "test"].reset_index(drop=True)
    pools = {"within_slide_test": test}
    if external_root is not None:
        from cellselect.external_data import load_external

        external = load_external(external_root)
        if file_sha256(Path(external_root) / "manifest.json") != protocol.get(
            "external_manifest_sha256"
        ):
            raise ValueError("External manifest changed")
        pools.update(
            {
                name: external.metadata.iloc[external.pool_rows(name)].reset_index(drop=True)
                for name in protocol["external_pools"]
            }
        )
    scores = {}
    for record in results["predictions"]:
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError("Saved prediction checksum changed")
        with np.load(path, allow_pickle=False) as values:
            if not np.array_equal(
                values["global_index"], pools[record["pool"]].global_index.to_numpy()
            ):
                raise ValueError("Saved predictions do not match evaluation rows")
            for policy in record["policies"]:
                scores[record["group"], record["pool"], policy, record["seed"]] = values[
                    "score"
                ].copy()
    comparisons = {
        "joint_minus_regression": ("joint", "regression"),
        "spo_plus_minus_regression": ("spo_plus", "regression"),
        "pg_central_minus_regression": ("pg_central", "regression"),
        "utility_minus_predictive_selector": ("utility_selector", "predictive_selector"),
        "regression_minus_class_means": ("regression", "class_means"),
    }
    rows = []
    for gkey, group in protocol["groups"].items():
        for pool_id, meta in pools.items():
            if pool_id != "within_slide_test" and group["train_fraction"] != 1:
                continue
            y, ids, fields = (
                meta[group["marker"]].to_numpy(),
                meta.global_index.to_numpy(),
                meta.fov_id.to_numpy(),
            )
            for name, (a, b) in comparisons.items():
                sa = np.stack([scores[gkey, pool_id, a, seed] for seed in group["seeds"]])
                sb = np.stack([scores[gkey, pool_id, b, seed] for seed in group["seeds"]])
                for q in protocol["fractions"]:
                    interval = paired_topk_bootstrap(
                        y, sa, sb, fields, q, n_bootstrap=repetitions, seed=20261022, ids=ids
                    )
                    bootstrap_seed = interval.pop("seed")
                    rows.append(
                        {
                            "group": gkey,
                            "pool": pool_id,
                            "marker": group["marker"],
                            "train_fraction": group["train_fraction"],
                            "comparison": name,
                            "fraction": q,
                            "bootstrap_seed": bootstrap_seed,
                            **interval,
                            "exchange_diagnostics": exchange_diagnostics(y, sa, sb, ids, q),
                        }
                    )
    result = {
        "protocol_sha256": file_sha256(protocol_file),
        "results_sha256": file_sha256(root / "results.json"),
        "comparisons": rows,
        "scope": "Per-comparison descriptive 95 percent paired field intervals, conditional on frozen models/choices; no retraining, selection or donor uncertainty; no family-wise significance claim.",
    }
    write_json(target, result)
    pd.DataFrame(rows).to_csv(root / "comparisons.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "evaluate", "uncertainty"])
    parser.add_argument("--data", default="data/prepared/bsccmnist_cd16_v1")
    parser.add_argument("--external", default="data/prepared/bsccmnist_batch1_all_v1")
    parser.add_argument("--runs", default="results/runs/extended")
    parser.add_argument("--spec", default="configs/extended.json")
    parser.add_argument("--protocol", default="results/extended_protocol.json")
    parser.add_argument("--output", default="results/extended")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu")
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze_study(args.data, args.runs, args.spec, args.protocol, args.external)
        print(
            json.dumps(
                {
                    "groups": list(result["groups"]),
                    "frozen_candidate_states": len(result["records"]),
                }
            )
        )
    elif args.stage == "evaluate":
        result = evaluate_study(
            args.data, args.runs, args.protocol, args.output, args.external, device=args.device
        )
        print(
            json.dumps(
                {
                    "metric_rows": len(result["metrics"]),
                    "prediction_files": len(result["predictions"]),
                }
            )
        )
    else:
        result = uncertainty(args.data, args.protocol, args.output, args.external, args.bootstrap)
        print(json.dumps({"paired_comparisons": len(result["comparisons"])}))


if __name__ == "__main__":
    main()
