"""Run or dry-run the fixed 48-configuration 2x2 experiment.

The default mode is dry-run. Real execution requires ``--execute`` and is
never allowed to redirect output or overwrite an existing result.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config
from ddos_tl.experiments.runner import dry_run, execute, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and print; never train")
    mode.add_argument("--execute", action="store_true", help="explicitly run the 48 experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    manifest = load_manifest()
    if args.execute:
        execute(cfg, manifest)
    else:
        dry_run(cfg, manifest)


if __name__ == "__main__":
    main()
