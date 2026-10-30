"""Plot the completed study and follow-up diagnostics from saved results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from summarize_extended_results import load_verified_inputs

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results" / "figures_extended"
COLORS = {
    "regression": "#0072B2",
    "joint": "#E69F00",
    "spo_plus": "#009E73",
    "pg_central": "#D55E00",
    "class_means": "#CC79A7",
}
LABELS = {
    "regression": "Regression",
    "joint": "Joint CE + MSE",
    "spo_plus": "SPO+",
    "pg_central": "Central PG",
    "class_means": "Class-mean score",
}


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, name):
    DEST.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(DEST / f"{name}.{extension}", bbox_inches="tight")
    plt.close(fig)


def readout_and_validation():
    probe = pd.DataFrame(read("results/readout_probe/test_results.json")["metrics"])
    budget = read("results/validation_budget_summary.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), layout="constrained")
    first = probe[probe.fraction == 0.1]
    names = ["ce", "mse", "joint1"]
    for offset, kind, label, color in [
        (-0.17, "native_score", "Original score", "#999999"),
        (0.17, "fresh_readout", "Fresh regression head", "#0072B2"),
    ]:
        values = [
            first[(first.family == name) & (first.kind == kind)].utility.to_numpy()
            for name in names
        ]
        axes[0].bar(
            np.arange(3) + offset,
            [x.mean() for x in values],
            width=0.32,
            yerr=[x.std(ddof=1) for x in values],
            capsize=3,
            color=color,
            label=label,
        )
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=["CE encoder", "MSE encoder", "Joint encoder"],
        ylabel="Top-10% mean CD16",
        title="A. Matched readouts on frozen encoders",
        ylim=(0, 760),
    )
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].text(
        0.01,
        -0.23,
        "Frozen encoders; bars show mean ± SD of 3 paired runs.",
        transform=axes[0].transAxes,
        fontsize=8,
    )
    paired = pd.DataFrame(budget["paired_comparisons"])
    subset = paired[paired.fraction == 0.1].sort_values("budget_fraction")
    x = [
        next(
            r["selection_cells_mean"]
            for r in budget["summaries"]
            if r["budget_fraction"] == b and r["fraction"] == 0.1 and r["criterion"] == "rmse"
        )
        for b in subset.budget_fraction
    ]
    lower = [r[0] for r in subset.split_quantiles_025_975]
    upper = [r[1] for r in subset.split_quantiles_025_975]
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].fill_between(x, lower, upper, color="#0072B2", alpha=0.12)
    axes[1].plot(x, subset["mean"], "o-", color="#0072B2")
    axes[1].set(
        xlabel="Cells available for validation selection",
        ylabel="Assessment utility difference",
        title="B. Utility selection minus RMSE selection",
    )
    axes[1].text(
        0.01,
        -0.23,
        "500 repeated field splits; shaded 2.5–97.5% quantiles (not a CI).",
        transform=axes[1].transAxes,
        fontsize=8,
    )
    save(fig, "readout_and_validation")


def completed_study():
    result_path = ROOT / "results/extended/results.json"
    if not result_path.exists():
        return
    _, results, uncertainty, _ = load_verified_inputs(
        ROOT / "results/extended_protocol.json", result_path.parent
    )
    frame = pd.DataFrame(results["metrics"])
    main = frame[(frame.pool == "within_slide_test") & (frame.fraction == 0.1)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for policy in COLORS:
        rows = main[(main.marker == "CD16") & (main.policy == policy)]
        stats = rows.groupby("train_fraction").utility.agg(["mean", "std"])
        axes[0].errorbar(
            stats.index * 100,
            stats["mean"],
            yerr=stats["std"],
            marker="o",
            capsize=3,
            color=COLORS[policy],
            label=LABELS[policy],
        )
    axes[0].set(
        xlabel="Training fields retained (%)",
        ylabel="Top-10% mean CD16",
        xticks=[25, 50, 100],
        title="A. Label-budget comparison",
    )
    axes[0].legend(fontsize=8)
    for index, marker in enumerate(["CD16", "CD45"]):
        rows = main[(main.marker == marker) & (main.train_fraction == 1)]
        baseline = rows[rows.policy == "regression"].sort_values("seed").utility.to_numpy()
        if np.any(baseline == 0):
            raise ValueError("Relative utility change is undefined for a zero regression baseline")
        # Normalize within marker for display only, never across candidate pools.
        for j, policy in enumerate(["joint", "spo_plus", "pg_central", "class_means"]):
            values = rows[rows.policy == policy].sort_values("seed").utility.to_numpy()
            relative = (values - baseline) / baseline * 100
            axes[1].errorbar(
                index + (j - 1.5) * 0.16,
                relative.mean(),
                yerr=relative.std(ddof=1),
                marker="o",
                capsize=3,
                color=COLORS[policy],
                label=LABELS[policy] if index == 0 else None,
            )
    axes[1].axhline(0, color="0.4", linewidth=0.8)
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["CD16", "CD45"],
        ylabel="Paired change from regression (%)",
        title="B. Full-cohort marker comparison",
    )
    axes[1].legend(fontsize=8)
    fig.supxlabel(
        "Mean ± SD across paired replicates; low-data replicates vary both subset and initialization.",
        fontsize=8,
    )
    save(fig, "objectives_and_label_budget")
    full = frame[(frame.pool == "within_slide_test") & (frame.train_fraction == 1)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for ax, marker in zip(axes, ["CD16", "CD45"]):
        for policy in COLORS:
            stats = (
                full[(full.marker == marker) & (full.policy == policy)]
                .groupby("fraction")
                .utility.agg(["mean", "std"])
            )
            ax.errorbar(
                stats.index * 100,
                stats["mean"],
                yerr=stats["std"],
                marker="o",
                capsize=3,
                color=COLORS[policy],
                label=LABELS[policy],
            )
        ax.set(
            xlabel="Cells selected (%)",
            ylabel=f"Mean measured {marker}",
            xticks=[5, 10, 20],
            title=marker,
        )
    axes[0].legend(fontsize=8)
    fig.supxlabel(
        "Models selected at the 10% validation budget; no retuning at 5% or 20%. Mean ± paired-run SD.",
        fontsize=8,
    )
    save(fig, "selection_budget_transfer")
    # Explain attained enrichment, separately from the noisy hindsight optimum.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.7), layout="constrained")
    policies = list(COLORS)
    short_labels = ["Regression", "Joint", "SPO+", "PG", "Class mean"]
    for ax, marker in zip(axes, ["CD16", "CD45"]):
        rows = main[(main.marker == marker) & (main.train_fraction == 1)].copy()
        rows["type_mix_gain"] = rows.type_matched_random_utility - rows.random_expected_utility
        for offset, column, label, color in [
            (-0.17, "type_mix_gain", "Type-mix contribution", "#999999"),
            (0.17, "within_type_enrichment", "Within-type contribution", "#0072B2"),
        ]:
            means = [rows[rows.policy == policy][column].mean() for policy in policies]
            ax.bar(np.arange(len(policies)) + offset, means, width=0.32, color=color, label=label)
        ax.axhline(0, color="0.4", linewidth=0.8)
        ax.set(
            xticks=np.arange(len(policies)),
            xticklabels=short_labels,
            ylabel=f"{marker} enrichment contribution",
            title=marker,
        )
        ax.set_ylim(top=1.28 * rows[["type_mix_gain", "within_type_enrichment"]].max().max())
        ax.tick_params(axis="x", labelsize=9, labelrotation=15)
    axes[0].legend(fontsize=8)
    fig.supxlabel(
        "The two terms sum to enrichment above random selection. True type labels enter diagnosis only.",
        fontsize=8,
    )
    save(fig, "achieved_enrichment")
    if (ROOT / "results/extended/uncertainty.json").exists():
        comparison = pd.DataFrame(uncertainty["comparisons"])
        comparison = comparison[
            (comparison.pool == "within_slide_test")
            & (comparison.train_fraction == 1)
            & (comparison.fraction == 0.1)
        ]
        contrasts = [
            "joint_minus_regression",
            "spo_plus_minus_regression",
            "pg_central_minus_regression",
            "utility_minus_predictive_selector",
        ]
        labels = [
            "Joint − regression",
            "SPO+ − regression",
            "PG − regression",
            "Utility − RMSE selection",
        ]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
        for ax, marker in zip(axes, ["CD16", "CD45"]):
            rows = comparison[comparison.marker == marker].set_index("comparison").loc[contrasts]
            estimate = rows.difference_a_minus_b.to_numpy()
            # Draw percentile intervals directly; they need not contain the point estimate.
            for i, (_, row) in enumerate(rows.iterrows()):
                ax.plot([row.ci_low, row.ci_high], [i, i], color="#0072B2")
            ax.scatter(estimate, np.arange(len(contrasts)), color="#0072B2", s=25)
            ax.axvline(0, color="0.4", linewidth=0.8)
            ax.set(
                yticks=np.arange(len(contrasts)),
                yticklabels=labels,
                xlabel="Top-10% utility difference",
                title=marker,
            )
            ax.invert_yaxis()
        fig.supxlabel(
            "95% paired-field intervals conditional on frozen models and choices; unadjusted per comparison.",
            fontsize=8,
        )
        save(fig, "paired_utility_differences")
    external = frame[(frame.pool != "within_slide_test") & (frame.fraction == 0.1)]
    if not external.empty:
        pools = sorted(external.pool.unique())
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
        for ax, marker in zip(axes, ["CD16", "CD45"]):
            for j, policy in enumerate(COLORS):
                means, deviations = [], []
                for pool in pools:
                    values = external[
                        (external.marker == marker)
                        & (external.pool == pool)
                        & (external.policy == policy)
                    ].enrichment_difference.to_numpy()
                    means.append(values.mean())
                    deviations.append(values.std(ddof=1))
                ax.errorbar(
                    np.arange(len(pools)) + (j - 2) * 0.14,
                    means,
                    yerr=deviations,
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    color=COLORS[policy],
                    label=LABELS[policy],
                )
            ax.axhline(0, color="0.4", linewidth=0.8)
            ax.set(
                xticks=np.arange(len(pools)),
                xticklabels=[f"Slide {i}" for i in range(len(pools))],
                ylabel=f"{marker} enrichment over random",
                title=f"Frozen-model external evaluation: {marker}",
            )
        axes[0].legend(fontsize=8)
        fig.supxlabel(
            "Slides are separate candidate pools; compare methods within a slide. Mean ± SD across 3 trained models.",
            fontsize=8,
        )
        save(fig, "external_acquisition")


def main():
    style()
    readout_and_validation()
    completed_study()
    print(f"Saved figures from completed results to {DEST}")


if __name__ == "__main__":
    main()
