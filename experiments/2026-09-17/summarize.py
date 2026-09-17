"""Recompute saved test metrics and summarize every preselected seed.

Run from the repository root: python experiments/2026-09-17/summarize.py
No model training or internal metric helpers are used here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44]
MODELS = ["LogReg", "XGBoost", "MLP-no-graph", "GCN"]


def strict_json(path):
    def reject(value):
        raise ValueError(f"Non-standard JSON number: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def independent_metrics(frame, model):
    y = frame["y_dir"].to_numpy()
    p = frame[model].to_numpy()
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError(f"Invalid probabilities for {model}")
    # Equal-width reliability bins, with p=0 in the first bin and p=1 in the last.
    bin_id = np.maximum(0, np.ceil(p * 10).astype(int) - 1)
    ece = sum(
        abs(y[bin_id == i].mean() - p[bin_id == i].mean()) * (bin_id == i).mean()
        for i in range(10) if (bin_id == i).any()
    )
    positive_hits, spreads = [], []
    for _, group in frame.groupby(level="date"):
        top = group[group[model] >= group[model].quantile(0.9)]
        bottom = group[group[model] <= group[model].quantile(0.1)]
        positive_hits.append((top["fwd_ret"] > 0).mean())
        spreads.append(top["fwd_ret"].mean() - bottom["fwd_ret"].mean())
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "top_hit": float(np.mean(positive_hits)),
        "long_short": float(np.mean(spreads)),
    }


rows, evidence = [], []
for seed in SEEDS:
    run = ROOT / f"seed-{seed}" / "outputs"
    result = strict_json(run / "reports/results.json")
    prediction_path = run / "reports/test_predictions.parquet"
    frame = pd.read_parquet(prediction_path)
    assert result["config"]["data"]["seed"] == seed
    assert result["config"]["label"]["horizon"] == 20
    assert result["config"]["split"]["purge_gap"] == 20
    assert result["config"]["data"]["n_stocks"] == 80
    assert result["config"]["data"]["n_days"] == 600
    assert not frame.index.duplicated().any()
    actual_checkpoint_hash = hashlib.sha256((run / "models/gcn.pt").read_bytes()).hexdigest()
    assert actual_checkpoint_hash == result["provenance"]["checkpoint_sha256"]
    for model in MODELS:
        stored = next(item for item in result["models"] if item["name"] == model)
        independent = independent_metrics(frame, model)
        assert len(frame) == stored["n_samples"] == 7600
        for metric, value in independent.items():
            np.testing.assert_allclose(value, stored[metric], rtol=0, atol=1e-12)
        rows.append({"seed": seed, "model": model, **independent})
    main_auc = next(item["auc"] for item in result["models"] if item["name"] == "GCN")
    assert result["edge_drop_ablation"]["0.0"] == main_auc
    evidence.append({
        "seed": seed,
        "test_rows": len(frame),
        "test_dates": frame.index.get_level_values("date").nunique(),
        "predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "provenance": result["provenance"],
        "edge_drop_auc": result["edge_drop_ablation"],
    })

assert len({row["provenance"]["data_sha256"]["returns.parquet"] for row in evidence}) == 3
assert len({row["provenance"]["source_revision"] for row in evidence}) == 1
assert len({row["provenance"]["source_sha256"] for row in evidence}) == 1
metrics = pd.DataFrame(rows)
metrics.to_csv(ROOT / "metrics.csv", index=False)
aggregate = {}
for model in MODELS:
    subset = metrics[metrics.model == model]
    aggregate[model] = {
        metric: {"mean": float(subset[metric].mean()), "sample_sd": float(subset[metric].std(ddof=1))}
        for metric in ["auc", "brier", "ece", "top_hit", "long_short"]
    }
pivot = metrics.pivot(index="seed", columns="model", values="auc")
delta = pivot["GCN"] - pivot["MLP-no-graph"]
summary = {
    "seeds": SEEDS,
    "data_source": "synthetic",
    "verification": "60 metrics independently recomputed from saved predictions; all within 1e-12; each zero-drop AUC equals main GCN AUC",
    "per_seed_metrics": rows,
    "aggregate": aggregate,
    "gcn_minus_mlp_auc": {"per_seed": {str(k): float(v) for k, v in delta.items()}, "mean": float(delta.mean()), "sample_sd": float(delta.std(ddof=1))},
    "evidence": evidence,
    "limitations": ["Three synthetic seeds, not real-market evidence or a significance test", "MLP and GCN architectures/training budgets differ", "No degree-preserving randomization experiment was run", "Network statistics are not model inputs; no_network_stat ablation is a no-op", "Overlapping 20-day horizon spreads are not a portfolio NAV", "Default 300-stock/1500-day training was not run"],
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

plt.rcParams.update({"font.size": 10, "figure.dpi": 160})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
colors = ["#7B8794", "#D69E36", "#3882A5", "#784BA0"]
for model, color in zip(MODELS, colors):
    axes[0].plot(SEEDS, pivot[model], marker="o", label=model, color=color)
axes[0].axhline(.5, linestyle="--", color="#9CA3AF", linewidth=1)
axes[0].set(xticks=SEEDS, xlabel="Synthetic seed", ylabel="Test AUC", title="All preselected seeds; 7,600 test rows per run", ylim=(.40, .57))
axes[0].legend(fontsize=9)
for item in evidence:
    pairs = sorted((float(k), v) for k, v in item["edge_drop_auc"].items())
    axes[1].plot([x for x, _ in pairs], [y for _, y in pairs], marker="o", label=f"Seed {item['seed']}")
axes[1].axhline(.5, linestyle="--", color="#9CA3AF", linewidth=1)
axes[1].set(xlabel="Fraction of test-graph edges removed", ylabel="Test AUC", title="Inference ablation on the same checkpoint", ylim=(.40, .57))
axes[1].legend(fontsize=9)
fig.suptitle("GCN research benchmark · synthetic data · 80 stocks × 600 dates", fontsize=13)
fig.tight_layout()
fig.savefig(ROOT / "benchmark.png")
plt.close(fig)
print(json.dumps({"aggregate": aggregate, "gcn_minus_mlp_auc": summary["gcn_minus_mlp_auc"], "verification": summary["verification"]}, indent=2))
