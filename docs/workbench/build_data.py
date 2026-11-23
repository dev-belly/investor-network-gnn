"""Build the offline browser snapshot from the repository's saved evidence."""

import argparse
import hashlib
import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def source(label, path):
    normalized = (ROOT / path).read_text(encoding="utf-8").encode("utf-8")
    return {
        "label": label,
        "url": f"{REPOSITORY}/blob/main/{path}",
        "sha256_lf": hashlib.sha256(normalized).hexdigest(),
    }


def fact(label, value, kind=None):
    return {"label": label, "value": value, "format": kind}


def column(key, title, kind=None):
    return {"key": key, "title": title, "format": kind}


def order(key, title, descending=True):
    return {"key": key, "title": title, "id": key, "descending": descending}


def metric(label, value, note):
    return {"label": label, "value": value, "note": note}


def table(title, columns, rows, *formats):
    return {
        "title": title,
        "table": {"columns": columns, "rows": rows, "formats": formats},
    }


def theme(accent, muted, base, glow):
    return {
        "accent": accent,
        "accent-soft": muted,
        "hero-base": base,
        "hero-glow": glow,
    }


REPOSITORY = "https://github.com/dev-belly/investor-network-gnn"
EXPERIMENT = "experiments/2026-09-28"


def build():
    summary = read_json(f"{EXPERIMENT}/summary.json")
    rows = summary["per_seed_metrics"]
    ablations = {
        seed: read_json(f"{EXPERIMENT}/seed-{seed}/outputs/reports/results.json")[
            "edge_drop_ablation"
        ]
        for seed in summary["seeds"]
    }
    paired = summary["gcn_minus_mlp_auc"]
    records = []
    for row in rows:
        seed = row["seed"]
        name = row["model"]
        draws = sorted(ablations[seed].items(), key=lambda pair: float(pair[0]))
        records.append(
            {
                "id": f"{name}-{seed}",
                "title": f"{name} · {seed}",
                "subtitle": "未来 20 日相对收益分类 · 合成 HS300 风格数据",
                "values": {
                    "seed": seed,
                    "model": name,
                    "auc": row["auc"],
                    "brier": row["brier"],
                    "ece": row["ece"],
                    "return": row["long_short"],
                },
                "facts": [
                    fact("ROC AUC", row["auc"], "fixed4"),
                    fact("Brier", row["brier"], "fixed4"),
                    fact("ECE", row["ece"], "fixed4"),
                    fact("Top-10 命中率", row["top_hit"], "percent"),
                    fact("多空十等分收益", row["long_short"], "percent"),
                    fact("本种子 GCN − MLP AUC", paired["per_seed"][str(seed)], "fixed4"),
                ],
                "charts": {
                    "ablation": [
                        {"label": f"删边 {float(level):.0%}", "value": auc} for level, auc in draws
                    ]
                },
                "sections": [
                    table(
                        "同一 GCN 模型的测试图删边",
                        ["移除边", "AUC"],
                        [[float(level), auc] for level, auc in draws],
                        "percent",
                        "fixed4",
                    )
                ],
            }
        )
    return {
        "repository": REPOSITORY,
        "slug": "investor-network-gnn",
        "project": "Investor Network GNN",
        "mark": "GN",
        "theme": theme("#624ead", "#f2efff", "#231d4c", "#5949a4"),
        "workspace": "图模型评估",
        "title": "图信号对照实验",
        "description": "四种模型、三个种子。比较排序和校准指标，再检验同一 GCN 对测试图删边的敏感性。",
        "dataKind": "合成股票面板 · 保存的样本外预测",
        "hero": metric("GCN − 无图 MLP 平均 AUC", f"{paired['mean']:+.4f}", "逐种子差值有正有负"),
        "metrics": [
            metric("独立种子", str(len(summary["seeds"])), "报告全部预先选定种子"),
            metric("模型与种子组合", str(len(records)), "每组保存逐行预测"),
            metric(
                "GCN 跨种子 AUC 标准差",
                f"{summary['aggregate']['GCN']['auc']['sample_sd']:.4f}",
                "样本标准差",
            ),
        ],
        "recordHeading": "样本外结果",
        "recordLabel": "模型 · 种子",
        "detailHeading": "模型证据",
        "searchPlaceholder": "搜索 GCN、MLP 或种子",
        "initialRecord": "GCN-42",
        "filters": [{"key": "seed", "title": "随机种子"}, {"key": "model", "title": "模型"}],
        "columns": [
            column("auc", "AUC", "fixed4"),
            column("brier", "Brier", "fixed4"),
            column("ece", "ECE", "fixed4"),
            column("return", "多空收益", "percent"),
        ],
        "orders": [order("auc", "AUC 最高"), order("brier", "Brier 最低", descending=False)],
        "views": [
            {
                "id": "auc",
                "label": "排序",
                "title": "AUC 高于或低于 0.50 的幅度",
                "metric": "auc",
                "reference": 0.5,
                "min": 0.44,
                "max": 0.56,
                "format": "fixed4",
                "limit": 12,
                "caption": "以 0.50 为中线显示全部可见组合。选择一条查看校准与收益指标。",
            },
            {
                "id": "brier",
                "label": "校准",
                "title": "Brier 概率误差",
                "metric": "brier",
                "format": "fixed4",
                "limit": 12,
                "caption": "数值越低，预测概率与二元结果之间的均方误差越小。",
            },
            {
                "id": "ablation",
                "label": "删边",
                "title": "所选种子的 GCN 删边实验",
                "mode": "detail",
                "reference": 0.5,
                "min": 0.44,
                "max": 0.56,
                "format": "fixed4",
                "caption": "固定所选种子的 GCN checkpoint，调整测试图的边保留比例。",
            },
        ],
        "sourceNote": "三种子比较中 GCN 对无图 MLP 的 AUC 差值有正有负。删边在测试阶段执行。输入为合成数据，采用一次按标签期限隔离的时间切分。",
        "sources": [
            source("多种子汇总与独立重算", f"{EXPERIMENT}/summary.json"),
            source("全部逐种子指标", f"{EXPERIMENT}/metrics.csv"),
            source("研究口径与复现步骤", "README.md"),
        ],
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the generated snapshot")
    args = parser.parse_args()
    document = build()
    records = document.pop("records")
    lines = ",\n".join(
        "    " + json.dumps(record, ensure_ascii=False, allow_nan=False) for record in records
    )
    header = json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2)
    result = (
        "// Generated by build_data.py from repository evidence.\n"
        "window.workbenchData = " + header[:-2] + ',\n  "records": [\n' + lines + "\n  ]\n};\n"
    )
    path = DIRECTORY / "data.js"
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != result:
            parser.exit(1, "Snapshot differs. Run build_data.py without --check.\n")
        print(f"Verified {len(records)} records")
    else:
        path.write_text(result, encoding="utf-8", newline="\n")
        print(f"Wrote {len(records)} records")


if __name__ == "__main__":
    main()
