"""Exact two-candidate SPO+ example; this is not a biological experiment.

A yields a with probability p and zero otherwise. B always yields b, 0<b<a.
The example specializes the known SPO+ consistency limitation outside the
distributional assumptions of Elmachtoub and Grigas (2022), pp. 19-20.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def validate(a, b, p):
    if not all(math.isfinite(x) for x in (a, b, p)) or not 0 < b < a or not 0 < p < 1:
        raise ValueError("Require finite a,b,p with 0 < b < a and 0 < p < 1")


def exact_risks(score_gaps, a=100.0, b=1.0, p=0.1):
    """Expected risks over the two possible outcomes, with no sampling."""
    validate(a, b, p)
    t = np.asarray(score_gaps, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError("Score gaps must be a finite vector")
    mean_a, mean_b = p * a, b
    chosen_utility = np.where(t >= 0, mean_a, mean_b)  # Stable tie favors A.
    return {
        "score_gap": t,
        "expected_spo_plus": p * np.maximum(a - b - 2 * t, 0) + (1 - p) * np.maximum(2 * t + b, 0),
        "expected_utility": chosen_utility,
        "expected_hindsight_regret": p * a + (1 - p) * b - chosen_utility,
        "expected_avoidable_regret": max(mean_a, mean_b) - chosen_utility,
        "selected_action": np.where(t >= 0, "A", "B"),
    }


def exact_summary(a=100.0, b=1.0, p=0.1):
    validate(a, b, p)
    lower, upper = -b / 2, (a - b) / 2
    means = {"A": p * a, "B": b}
    optimal = max(means.values())
    if p < 0.5:
        interval, choices = [lower, lower], ["B"]
    elif p > 0.5:
        interval, choices = [upper, upper], ["A"]
    else:
        interval, choices = [lower, upper], ["A", "B"]
    return {
        "role": "Exact analytical illustration, not a BSCCM result or a new general theorem",
        "parameters": {"a": a, "b": b, "p": p},
        "outcomes": [{"probability": p, "A": a, "B": b}, {"probability": 1 - p, "A": 0.0, "B": b}],
        "expected_outcomes": means,
        "mean_optimal_choices": [action for action, value in means.items() if value == optimal],
        "spo_minimizer_score_gap_interval": interval,
        "spo_minimizer_choices": choices,
        "minimum_expected_spo_plus": a * min(p, 1 - p),
        "avoidable_regret_at_spo_minimizers": {
            action: optimal - means[action] for action in choices
        },
        "expected_hindsight_oracle_utility": p * a + (1 - p) * b,
        "irreducible_hindsight_gap": p * a + (1 - p) * b - optimal,
        "score_tie_rule": "Select A when score_gap=0",
        "expected_spo_formula": "p*max(a-b-2*t,0)+(1-p)*max(2*t+b,0)",
        "piecewise_slopes": [-2 * p, 2 * (1 - 2 * p), 2 * (1 - p)],
        "breakpoints": [lower, upper],
        "assumption_note": "The distribution is discrete and asymmetric. This illustrates why SPO+ consistency cannot be invoked without its assumptions. No universal PG consistency is asserted.",
        "source": {
            "title": "Smart Predict, then Optimize",
            "authors": ["Adam N. Elmachtoub", "Paul Grigas"],
            "year": 2022,
            "doi": "10.1287/mnsc.2020.3922",
            "locations": "Assumption 1/Theorem 1 p.19; Proposition 6 and discussion p.20",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=float, default=100.0)
    parser.add_argument("--b", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("results/surrogate_example"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    summary = exact_summary(args.a, args.b, args.p)
    lower, upper = summary["breakpoints"]
    gaps = np.unique(
        np.r_[np.linspace(lower - 0.1 * args.a, upper + 0.1 * args.a, 501), lower, 0.0, upper]
    )
    values = exact_risks(gaps, args.a, args.b, args.p)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "exact_risks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(values.keys())
        writer.writerows(zip(*values.values()))
    summary["curve_file"] = "exact_risks.csv"
    summary["curve_points"] = len(gaps)
    if args.plot:
        import matplotlib.pyplot as plt

        minimum_interval = summary["spo_minimizer_score_gap_interval"]
        lo, hi = min(0.0, minimum_interval[0]), max(0.0, minimum_interval[1])
        padding = max(0.02 * args.a, 0.2 * (hi - lo))
        plot_gaps = np.unique(
            np.r_[np.linspace(lo - padding, hi + padding, 501), 0.0, minimum_interval]
        )
        plotted = exact_risks(plot_gaps, args.a, args.b, args.p)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(plot_gaps, plotted["expected_spo_plus"], label="Expected SPO+ loss")
        ax.step(
            plot_gaps,
            plotted["expected_hindsight_regret"],
            where="post",
            label="Expected hindsight regret",
        )
        ax.step(
            plot_gaps,
            plotted["expected_avoidable_regret"],
            where="post",
            label="Avoidable decision regret",
            linestyle="--",
        )
        ax.axvline(0, color="0.6", linewidth=0.8)
        if minimum_interval[0] == minimum_interval[1]:
            ax.plot(minimum_interval[0], summary["minimum_expected_spo_plus"], "o", color="C0")
        ax.set(
            xlabel="Predicted score gap (A minus B)",
            ylabel="Expected loss",
            title=f"Exact two-candidate example: a={args.a:g}, b={args.b:g}, p={args.p:g}",
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.output / "exact_risks.png", dpi=180)
        plt.close(fig)
        summary["figure_file"] = "exact_risks.png"
        summary["figure_score_gap_range"] = [lo - padding, hi + padding]
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in [
                    "expected_outcomes",
                    "spo_minimizer_score_gap_interval",
                    "spo_minimizer_choices",
                    "avoidable_regret_at_spo_minimizers",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
