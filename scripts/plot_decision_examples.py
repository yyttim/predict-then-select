"""Plot paired score and stopping-rule comparisons on the external pool."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_examples(results, output):
    results, output = Path(results), Path(output)
    primary = pd.read_csv(results / "external/tables/metrics.csv", float_precision="round_trip")
    stopping = pd.read_csv(
        results / "stopping/external/tables/metrics.csv", float_precision="round_trip"
    )

    def subset(frame, method, fraction, score):
        rows = (
            frame[
                (frame.method == method)
                & (frame.train_fraction == fraction)
                & (frame.selected_for == "purity")
                & (frame.score == score)
                & np.isclose(frame.selection_fraction, 0.1)
            ]
            .set_index("seed")
            .sort_index()
        )
        if rows.index.tolist() != [16, 26, 36] or not rows.n_cells.eq(27857).all():
            raise ValueError("Expected three paired runs on the complete external pool")
        return rows

    mean = subset(primary, "joint", 1.0, "mean")
    probability = subset(primary, "joint", 1.0, "hit_logit")
    utility_stop = subset(primary, "bce", 0.25, "hit_logit")
    loss_stop = subset(stopping, "bce", 0.25, "hit_logit")
    if not mean[["weight", "step"]].equals(probability[["weight", "step"]]):
        raise ValueError("The two joint scores must use the same checkpoints")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5), sharey=True)
    comparisons = [
        (mean, probability, "mean_marker", "Mean score", "Probability score"),
        (utility_stop, loss_stop, "log_loss", "Utility stopping", "Log-loss stopping"),
    ]
    for ax, (before, after, x, first, second) in zip(axes, comparisons):
        for seed in before.index:
            ax.plot(
                [before.loc[seed, x], after.loc[seed, x]],
                [100 * before.loc[seed, "purity"], 100 * after.loc[seed, "purity"]],
                color="0.72",
                linewidth=0.9,
                zorder=1,
            )
        ax.scatter(
            before[x], before.purity * 100, marker="o", s=25, color="#28628f", label=first, zorder=2
        )
        ax.scatter(
            after[x], after.purity * 100, marker="s", s=25, color="#ba722e", label=second, zorder=2
        )
        ax.set_ylim(5.8, 7.12)
        ax.set_yticks([6, 6.25, 6.5, 6.75, 7])
        ax.grid(axis="y", color="0.92", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_title("A  Score choice", loc="left")
    axes[0].set(xlabel="Mean selected CD16 (a.u.)", ylabel="Marker-high purity (%)")
    axes[0].legend(frameon=False, loc="lower left")
    axes[1].set_title("B  Stopping criterion", loc="left")
    axes[1].set_xlabel("Log loss (lower is better)")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.6, w_pad=1.4)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"decision_examples.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    plot_examples(args.results, args.output)
