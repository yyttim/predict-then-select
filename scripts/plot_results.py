"""Plot the frozen core evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/core/results.json")
    parser.add_argument("--protocol", default="results/core_protocol.json")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text())
    protocol = json.loads(Path(args.protocol).read_text())
    if results["protocol_sha256"] != hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest():
        raise ValueError("Results and frozen protocol do not match")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    table = pd.DataFrame(results["metrics"])
    primary = table[np.isclose(table.fraction, protocol["primary_fraction"])]
    joint = primary[primary["mode"] == "joint"]
    means = joint.groupby("lambda_class").mean(numeric_only=True).sort_index()
    spread = joint.groupby("lambda_class").std(numeric_only=True).fillna(0).sort_index()
    weights, x = means.index.to_numpy(), np.arange(len(means))
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, len(weights)))
    baseline = primary[primary["mode"] == "classifier"].mean(numeric_only=True)

    def save(fig, name):
        fig.savefig(output / f"{name}.pdf")
        fig.savefig(output / f"{name}.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), layout="constrained")
    fig.suptitle(f"Fixed epoch {results['reference_epoch']} · mean ± training-seed SD", fontsize=10)
    for ax, metric, title in zip(
        axes, ["rmse", "utility"], ["Marker prediction", "Cell selection"]
    ):
        ax.errorbar(x, means[metric], yerr=spread[metric], color="#275d8c", marker="o", capsize=3)
        ax.axhline(
            baseline[metric], color="0.5", linestyle="--", linewidth=1, label="Class-mean baseline"
        )
        ax.set(
            xticks=x,
            xticklabels=[f"{v:g}" for v in weights],
            xlabel=r"Classification weight $\lambda$",
            title=title,
        )
        ax.set_ylabel(
            "CD16 RMSE"
            if metric == "rmse"
            else f"Mean measured CD16 in top {100 * protocol['primary_fraction']:g}%"
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "prediction_and_selection")
    selected = pd.DataFrame(results["selected_metrics"])
    selected = selected[np.isclose(selected.fraction, protocol["primary_fraction"])]
    policies = ["rmse_selector", "utility_selector", "regression_baseline", "class_means_baseline"]
    labels = ["RMSE-selected", "Utility-selected", "Regression", "Class means"]
    selected_means = selected.groupby("policy").mean(numeric_only=True).loc[policies]
    selected_std = selected.groupby("policy").std(numeric_only=True).fillna(0).loc[policies]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), layout="constrained")
    fig.suptitle("Validation-selected policies · mean ± training-seed SD", fontsize=10)
    for ax, metric, ylabel in zip(axes, ["rmse", "utility"], ["CD16 RMSE", "Mean CD16 in top 10%"]):
        ax.errorbar(
            range(4),
            selected_means[metric],
            yerr=selected_std[metric],
            fmt="o",
            color="#275d8c",
            capsize=4,
            markersize=6,
        )
        ax.set(xticks=range(4), xticklabels=labels, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.15)
    save(fig, "validation_selected_policies")
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    gx = np.arange(len(weights) + 1)
    within = np.append(means.within_type_gap.to_numpy(), baseline.within_type_gap)
    allocation = np.append(means.type_allocation_gap.to_numpy(), baseline.type_allocation_gap)
    ax.bar(gx, within, color="#356b94", label="Within-type selection gap")
    ax.bar(gx, allocation, bottom=within, color="#a9bdcf", label="Type-allocation gap")
    ax.set(
        xticks=gx,
        xticklabels=[f"{v:g}" for v in weights] + ["Class\nmeans"],
        xlabel=r"Classification weight $\lambda$ / baseline",
        ylabel="Mean CD16 gap to hindsight optimum",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "selection_gap")
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    for i, weight in enumerate(weights):
        row = means.loc[weight]
        ax.annotate(
            "",
            xy=(row.calibrated_rmse, row.utility),
            xytext=(row.rmse, row.utility),
            arrowprops={"arrowstyle": "->", "color": colors[i], "lw": 1.3},
        )
        ax.plot(row.rmse, row.utility, "o", color=colors[i], label=rf"$\lambda={weight:g}$")
        ax.plot(row.calibrated_rmse, row.utility, "x", color=colors[i])
    ax.set(xlabel="CD16 RMSE (raw → validation-calibrated)", ylabel="Selection utility")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "calibration_control")
    fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
    for i, weight in enumerate(weights):
        sub = table[(table["mode"] == "joint") & (table.lambda_class == weight)]
        group = sub.groupby("fraction").utility.agg(["mean", "std"]).fillna(0).sort_index()
        ax.errorbar(
            100 * group.index.to_numpy(),
            group["mean"],
            yerr=group["std"],
            marker="o",
            capsize=2,
            color=colors[i],
            label=rf"$\lambda={weight:g}$",
        )
    ax.set(
        xlabel="Selected fraction (%)",
        ylabel="Mean measured CD16 of selected cells",
        xticks=[100 * q for q in protocol["fractions"]],
    )
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "selection_fraction")
    print(
        f"Saved five PDF/PNG figures to {output}; weight frontiers use epoch {results['reference_epoch']}"
    )


if __name__ == "__main__":
    main()
