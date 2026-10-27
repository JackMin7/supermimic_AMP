"""Inspect scaffold defaults; this entry point does not launch training."""
import argparse
import json
from dataclasses import asdict
from .config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.print_config:
        print(json.dumps({"status": "scaffold_only", **asdict(ExperimentConfig())}, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
