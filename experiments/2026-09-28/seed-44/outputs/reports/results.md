# Investor Network GCN — Experiment Report

## 1. Test-set metrics

| Model | n | AUC | Brier | ECE | Top-10 hit | Long-Short |
|---|---|---|---|---|---|---|
| LogReg | 7600 | 0.4874 | 0.2583 | 0.0699 | 0.5697 | -0.0222 |
| XGBoost | 7600 | 0.5073 | 0.2987 | 0.1974 | 0.6118 | -0.0034 |
| MLP-no-graph | 7600 | 0.4802 | 0.2618 | 0.0763 | 0.5829 | -0.0199 |
| GCN | 7600 | 0.5181 | 0.2497 | 0.0184 | 0.5500 | 0.0080 |

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
| 0.00 | 0.5181 |
| 0.25 | 0.5147 |
| 0.50 | 0.5122 |
| 0.75 | 0.5128 |
| 0.90 | 0.5048 |

## 4. AUC by market regime

| Model | bull | bear | sideways |
|---|---|---|---|
| LogReg | 0.3881 | n/a | 0.5933 |
| XGBoost | 0.4506 | n/a | 0.5690 |
| MLP-no-graph | 0.4369 | n/a | 0.5409 |
| GCN | 0.5722 | n/a | 0.4585 |

## 5. Per-industry highlights

Top industries by GCN AUC:
- 农林牧渔: AUC=0.8851, n=190
- 家用电器: AUC=0.6816, n=475
- 计算机: AUC=0.6755, n=95
- 公用事业: AUC=0.6636, n=475
- 医药生物: AUC=0.6347, n=190

Worst industries by GCN AUC:
- 传媒: AUC=0.2944, n=285
- 电力设备: AUC=0.3316, n=380
- 综合: AUC=0.3476, n=380
- 建筑材料: AUC=0.3676, n=95
- 电子: AUC=0.3935, n=285