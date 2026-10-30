import argparse
from pathlib import Path

from cellselect.data import load_data
from cellselect.evaluation import evaluate_pool, freeze_selection


def main():
    parser = argparse.ArgumentParser(
        description="Select models on validation data, then evaluate fixed test policies."
    )
    parser.add_argument("action", choices=("freeze", "test", "external"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--study", type=Path, default=Path("results/study"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    data = load_data(args.data)
    if args.action == "freeze":
        choices = freeze_selection(data, args.study)
        for choice in choices["selections"]:
            print(
                f"{choice['train_fraction']:g} {choice['selected_for']} {choice['method']}: "
                f"step {choice['step']}, weight {choice['weight']:g}"
            )
    else:
        evaluate_pool(data, args.study, args.device, pool=args.action)


if __name__ == "__main__":
    main()
