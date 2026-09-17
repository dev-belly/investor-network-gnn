# Investor Network GCN — Experiment Report

## 1. Test-set metrics

| Model | n | AUC | Brier | ECE | Top-10 hit | Long-Short |
|---|---|---|---|---|---|---|
| LogReg | 7600 | 0.4874 | 0.2583 | 0.0699 | 0.5697 | -0.0222 |
| XGBoost | 7600 | 0.5073 | 0.2987 | 0.1974 | 0.6118 | -0.0034 |
| MLP-no-graph | 7600 | 0.4802 | 0.2618 | 0.0763 | 0.5829 | -0.0199 |
| GCN | 7600 | 0.5162 | 0.2498 | 0.0112 | 0.5461 | 0.0124 |

## 2. Feature ablation (XGBoost)

| Setting | n_feat | AUC | Brier |
|---|---|---|---|
| all | 23 | 0.5073 | 0.2987 |
| no_return_vol | 13 | 0.5181 | 0.3068 |
| no_fundamental | 14 | 0.4979 | 0.2738 |
| no_momentum | 19 | 0.5068 | 0.2923 |
| no_network_stat | 23 | 0.5073 | 0.2987 |

## 3. Edge-drop ablation (GCN)

| drop_rate | AUC |
|---|---|
| 0.00 | 0.5162 |
| 0.25 | 0.5106 |
| 0.50 | 0.5102 |
| 0.75 | 0.5005 |
| 0.90 | 0.4994 |

## 4. AUC by market regime

| Model | bull | bear | sideways |
|---|---|---|---|
| LogReg | 0.3881 | n/a | 0.5933 |
| XGBoost | 0.4506 | n/a | 0.5690 |
| MLP-no-graph | 0.4369 | n/a | 0.5409 |
| GCN | 0.5409 | n/a | 0.4900 |

## 5. Per-industry highlights

Top industries by GCN AUC:
- 农林牧渔: AUC=0.8676, n=190
- 计算机: AUC=0.7823, n=95
- 医药生物: AUC=0.7183, n=190
- 纺织服饰: AUC=0.6744, n=285
- 食品饮料: AUC=0.6731, n=95

Worst industries by GCN AUC:
- 传媒: AUC=0.3399, n=285
- 机械设备: AUC=0.3660, n=285
- 综合: AUC=0.3737, n=380
- 电力设备: AUC=0.3772, n=380
- 银行: AUC=0.3898, n=475