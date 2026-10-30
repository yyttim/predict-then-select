import argparse
from pathlib import Path

from cellselect.reporting import summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize the frozen cell-selection study.")
    parser.add_argument("--study", type=Path, default=Path("results/study"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pool", choices=("test", "external"), default="test")
    args = parser.parse_args()
    output = args.output or (Path("results") if args.pool == "test" else Path("results/external"))
    summarize(args.study, output, pool=args.pool)
