"""Tables, paired comparisons, and figures for the fixed test policies."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import paired_field_bootstrap
from .study import file_hash, save_json


def summarize(study_root, output, draws=None, pool="test"):
    if pool not in {"test", "external"}:
        raise ValueError("Choose the main test pool or the external pool")
    root, output = Path(study_root), Path(output)
    existing = output / "summary.json"
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous["pool"] != pool or previous["selection_sha256"] != file_hash(
            root / "selection.json"
        ):
            raise FileExistsError(
                "This output contains another pool or study; choose a new directory"
            )
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    spec = json.loads((root / "study.json").read_text(encoding="utf-8"))
    choices = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    pool_root = root / pool
    evaluation = json.loads((pool_root / "evaluation.json").read_text(encoding="utf-8"))
    if evaluation["status"] != "complete" or evaluation["selection_sha256"] != file_hash(
        root / "selection.json"
    ):
        raise ValueError("Incomplete test evaluation or changed selection")
    if evaluation["metrics_sha256"] != file_hash(pool_root / "metrics.csv"):
        raise ValueError("Test metrics changed after evaluation")
    if evaluation["cells_sha256"] != file_hash(pool_root / "cells.csv"):
        raise ValueError("Test cell identities or targets changed")
    metrics = pd.read_csv(pool_root / "metrics.csv", float_precision="round_trip")
    cells = pd.read_csv(pool_root / "cells.csv", float_precision="round_trip")
    predictions = {}
    for entry in evaluation["prediction_files"]:
        path = pool_root / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise ValueError("Test predictions changed")
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["global_index"], cells.global_index.to_numpy()):
                raise ValueError("Test cell IDs differ")
            predictions[entry["checkpoint"]] = {
                k: data[k] for k in data.files if k != "global_index"
            }
    groups = [
        "train_fraction",
        "selected_for",
        "method",
        "weight",
        "step",
        "score",
        "primary_score",
        "selection_fraction",
    ]
    columns = [c for c in ("mean_marker", "purity", "rmse", "brier", "log_loss") if c in metrics]
    grouped = metrics.groupby(groups)[columns].agg(["mean", "std"])
    grouped.columns = [
        f"{metric}_{'sd' if statistic == 'std' else statistic}"
        for metric, statistic in grouped.columns
    ]
    summary = grouped.reset_index()
    metrics.to_csv(tables / "metrics.csv", index=False)
    head_keys = ["train_fraction", "selected_for", "method", "weight", "step", "seed"]
    head_metrics = [column for column in ("rmse", "brier", "log_loss") if column in metrics]
    metrics[head_keys + head_metrics].drop_duplicates().to_csv(
        tables / "head_diagnostics.csv", index=False
    )
    summary.to_csv(tables / "summary.csv", index=False)
    pd.DataFrame(
        [{k: v for k, v in c.items() if k != "states"} for c in choices["selections"]]
    ).to_csv(tables / "model_choices.csv", index=False)
    metrics[
        [
            "selection_fraction",
            "n_cells",
            "n_selected",
            "n_high",
            "random_mean",
            "random_purity",
            "oracle_mean",
            "oracle_purity",
        ]
    ].drop_duplicates().to_csv(tables / "candidate_pool.csv", index=False)

    selected = {}
    for choice in choices["selections"]:
        key = (choice["train_fraction"], choice["selected_for"], choice["method"])
        state_predictions = [predictions[state["checkpoint"]] for state in choice["states"]]
        selected[key] = {
            score: np.stack([item[score] for item in state_predictions])
            for score in state_predictions[0]
        }
    settings = spec["settings"]
    common = {
        "target": cells.target.to_numpy(),
        "ids": cells.global_index.to_numpy(),
        "fields": cells.fov_id.to_numpy(),
        "threshold": choices["threshold"],
        "fractions": settings["selection_fractions"],
        "draws": draws or settings["bootstrap_draws"],
        "seed": settings["bootstrap_seed"],
    }

    def score(fraction, purpose, method):
        name = (
            "mean" if method == "mse" or (method == "joint" and purpose == "mean") else "hit_logit"
        )
        return selected[(fraction, purpose, method)][name]

    comparisons = []
    pairs = (("bce", "mse"), ("joint", "mse"), ("joint", "bce"))
    for fraction in settings["train_fractions"]:
        for purpose in ("mean", "purity"):
            for left, right in pairs:
                rows = paired_field_bootstrap(
                    scores_a=score(fraction, purpose, left),
                    scores_b=score(fraction, purpose, right),
                    **common,
                )
                comparisons.extend(
                    {
                        "train_fraction": fraction,
                        "selected_for": purpose,
                        "comparison": f"{left}_minus_{right}",
                        **row,
                    }
                    for row in rows
                )
            heads = selected[(fraction, purpose, "joint")]
            rows = paired_field_bootstrap(
                scores_a=heads["hit_logit"], scores_b=heads["mean"], **common
            )
            comparisons.extend(
                {
                    "train_fraction": fraction,
                    "selected_for": purpose,
                    "comparison": "joint_hit_minus_joint_mean",
                    **row,
                }
                for row in rows
            )
            print(f"Paired field intervals: {fraction:g}, {purpose}", flush=True)
    paired = pd.DataFrame(comparisons)
    paired.to_csv(tables / "paired_comparisons.csv", index=False)
    interactions = []
    for purpose in ("mean", "purity"):
        for left, right in pairs:
            rows = paired_field_bootstrap(
                scores_a=score(0.25, purpose, left),
                scores_b=score(0.25, purpose, right),
                subtract_pair=(score(1.0, purpose, left), score(1.0, purpose, right)),
                **common,
            )
            interactions.extend(
                {
                    "selected_for": purpose,
                    "comparison": f"{left}_minus_{right}",
                    "contrast": "quarter_difference_minus_full_difference",
                    **row,
                }
                for row in rows
            )
    pd.DataFrame(interactions).to_csv(tables / "budget_interactions.csv", index=False)
    pool_label = "Main test" if pool == "test" else "External cohort"
    rule = (
        "predictive-loss selection"
        if "primary_selection_sha256" in choices
        else "utility selection"
    )
    plot_results(summary, paired, figures, settings["primary_fraction"], f"{pool_label}, {rule}")
    save_json(
        output / "summary.json",
        {
            "study_sha256": file_hash(root / "study.json"),
            "selection_sha256": file_hash(root / "selection.json"),
            "threshold": choices["threshold"],
            "pool": pool,
            "cells": evaluation["n_cells"],
            "fields": evaluation["n_fields"],
            "training_runs": len(spec["runs"]),
            "metric_rows": len(metrics),
            "paired_interval_rows": len(paired),
            "bootstrap_draws": common["draws"],
            "interval_scope": "Conditional on fixed trained states, validation choices, cutoff and data split",
        },
    )
    return summary, paired


def plot_results(summary, paired, output, q, title=""):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    colors = {"mse": "#28628f", "bce": "#b86832", "joint": "#30826e"}
    labels = {"mse": "Regression", "bce": "Threshold classification", "joint": "Joint training"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, purpose, metric, scale, ylabel in zip(
        axes,
        ("purity", "mean"),
        ("purity", "mean_marker"),
        (100, 1),
        ("Marker-high purity (%)", "Mean selected CD16 (a.u.)"),
    ):
        data = summary[
            (summary.selected_for == purpose)
            & summary.primary_score
            & np.isclose(summary.selection_fraction, q)
        ]
        for method, color in colors.items():
            rows = data[data.method == method].sort_values("train_fraction")
            ax.errorbar(
                rows.train_fraction * 100,
                rows[f"{metric}_mean"] * scale,
                yerr=rows[f"{metric}_sd"] * scale,
                marker="o",
                capsize=3,
                linewidth=1.5,
                color=color,
                label=labels[method],
            )
        ax.set(
            xlabel="Training fields retained (%)", ylabel=ylabel, xticks=[25, 100], xlim=(18, 107)
        )
        ax.set_title("Select for purity" if purpose == "purity" else "Select for mean abundance")
        ax.grid(axis="y", color="0.92")
    handles, names = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{title} | top {q:.0%} | mean ± seed SD", fontsize=11)
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"objectives.{ext}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    pair_names = {
        "bce_minus_mse": "Classification - regression",
        "joint_minus_mse": "Joint - regression",
        "joint_minus_bce": "Joint - classification",
    }
    for ax, purpose, metric, scale, xlabel in zip(
        axes,
        ("purity", "mean"),
        ("purity", "mean_marker"),
        (100, 1),
        ("Purity difference (percentage points)", "Mean CD16 difference (a.u.)"),
    ):
        data = paired[
            (paired.selected_for == purpose)
            & (paired.metric == metric)
            & np.isclose(paired.selection_fraction, q)
        ]
        for i, (fraction, color, marker) in enumerate(
            ((0.25, "#b86832", "o"), (1.0, "#28628f", "s"))
        ):
            rows = (
                data[data.train_fraction == fraction].set_index("comparison").loc[list(pair_names)]
            )
            y = np.arange(len(pair_names)) + (i - 0.5) * 0.18
            x = rows.difference.to_numpy() * scale
            ax.hlines(
                y, rows.ci_low.to_numpy() * scale, rows.ci_high.to_numpy() * scale, color=color
            )
            ax.plot(x, y, marker, color=color, label=f"{fraction:.0%} training fields")
        ax.axvline(0, color="0.6", linewidth=0.8)
        ax.set(yticks=range(len(pair_names)), yticklabels=list(pair_names.values()), xlabel=xlabel)
        ax.set_ylim(2.35, -0.35)
        ax.tick_params(axis="y", labelleft=ax is axes[0])
        ax.grid(axis="x", color="0.92")
    handles, names = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{title} | top {q:.0%} | conditional 95% field intervals", fontsize=11)
    fig.legend(handles, names, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"paired_effects.{ext}", dpi=200)
    plt.close(fig)
