import argparse
from pathlib import Path

import pandas as pd

from src.config import REPORTS
from src.metrics import METRICS, load_log, load_results, run_all, _save_results
from src.miners import discover_inductive, save_petri_net_pnml, save_petri_net_svg

THRESHOLDS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
OUT_DIR = REPORTS / "inductive"
SUMMARY_CSV = OUT_DIR / "threshold_summary.csv"


def _tag(threshold: float) -> str:
    return f"{round(threshold * 100):02d}"


def metrics_path(threshold: float, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"threshold_{_tag(threshold)}_metrics.json"


""" Discovers the inductive model for one noise threshold, saves it (PNML + SVG)
    to out_dir and computes the quality metrics with metrics.run_all. The results
    are written to threshold_xx_metrics.json / .csv in out_dir.
"""
def evaluate_threshold(threshold: float, metrics=None, out_dir: Path = OUT_DIR) -> dict:
    name = f"inductive_{_tag(threshold)}"
    path = metrics_path(threshold, out_dir)

    print(f"[{name}] discovering model (noise_threshold={threshold}) ...", flush=True)
    model = discover_inductive(load_log(), noise_threshold=threshold)
    save_petri_net_pnml(model, name, out_dir)
    save_petri_net_svg(model, name, out_dir)

    results = run_all(miners=[name], metrics=metrics, path=path,
                      model_dir=out_dir,
                      skip_soundness=True,  # inductive nets are sound by construction
                      skip_alignments=True)  # alignments only for the initial miners
    results[name]["noise_threshold"] = threshold
    _save_results(results, path)
    return results


""" Combines the per-threshold results into one table (one column per threshold). """
def summary(thresholds=THRESHOLDS, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    frames = [load_results(metrics_path(t, out_dir)) for t in thresholds
              if metrics_path(t, out_dir).exists()]
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def run_sweep(thresholds=THRESHOLDS, metrics=None, out_dir: Path = OUT_DIR) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in thresholds:
        evaluate_threshold(t, metrics, out_dir)
    df = summary(thresholds, out_dir)
    df.to_csv(out_dir / SUMMARY_CSV.name)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the inductive miner for different noise thresholds.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(THRESHOLDS))
    parser.add_argument("--metrics", nargs="+", default=None, choices=list(METRICS))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(run_sweep(args.thresholds, args.metrics, args.out_dir))
