# Investor Network GCN — Experiment Report

## 1. Test-set metrics

| Model | n | AUC | Brier | ECE | Top-10 hit | Long-Short |
|---|---|---|---|---|---|---|
| LogReg | 7600 | 0.5078 | 0.2522 | 0.0395 | 0.5566 | 0.0059 |
| XGBoost | 7600 | 0.4520 | 0.3046 | 0.2004 | 0.4987 | -0.0078 |
| MLP-no-graph | 7600 | 0.4989 | 0.2524 | 0.0517 | 0.5263 | 0.0037 |
| GCN | 7600 | 0.4439 | 0.2539 | 0.0343 | 0.5447 | -0.0073 |

## 2. Feature ablation (XGBoost)

| Setting | n_feat | AUC | Brier |
|---|---|---|---|
| all | 23 | 0.4520 | 0.3046 |
| no_return_vol | 13 | 0.4534 | 0.3080 |
| no_fundamental | 14 | 0.4772 | 0.2878 |
| no_momentum | 19 | 0.4397 | 0.3065 |
| no_network_stat | 23 | 0.4520 | 0.3046 |

## 3. Edge-drop ablation (GCN)

| drop_rate | AUC |
|---|---|
| 0.00 | 0.4439 |
| 0.25 | 0.4496 |
| 0.50 | 0.4394 |
| 0.75 | 0.4446 |
| 0.90 | 0.4355 |

## 4. AUC by market regime

| Model | bull | bear | sideways |
|---|---|---|---|
| LogReg | 0.5378 | 0.4565 | 0.4971 |
| XGBoost | 0.4498 | 0.4325 | 0.4619 |
| MLP-no-graph | 0.5043 | 0.4557 | 0.5136 |
| GCN | 0.5302 | 0.3887 | 0.3863 |

## 5. Per-industry highlights

Top industries by GCN AUC:
- 钢铁: AUC=0.8902, n=95
- 建筑装饰: AUC=0.6756, n=190
- 食品饮料: AUC=0.6397, n=95
- 通信: AUC=0.6279, n=285
- 传媒: AUC=0.6205, n=475

Worst industries by GCN AUC:
- 医药生物: AUC=0.1097, n=190
- 基础化工: AUC=0.2356, n=285
- 家用电器: AUC=0.2400, n=95
- 有色金属: AUC=0.2746, n=570
- 公用事业: AUC=0.3242, n=380