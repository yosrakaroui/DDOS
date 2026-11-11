"""Génère de faux CSV au format CIC pour tester tout le pipeline (smoke test).

Usage : python scripts/make_synthetic_data.py --config configs/synthetic.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ddos_tl.config import load_config, resolve
from ddos_tl.data.synthetic import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    generate(resolve(cfg["paths"]["source_raw"]), resolve(cfg["paths"]["target_raw"]))
    print("Données synthétiques générées dans", resolve(cfg["paths"]["target_raw"]).parent)
