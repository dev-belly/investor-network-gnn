# Investor Network GCN — Experiment Report

## 1. Test-set metrics

| Model | n | AUC | Brier | ECE | Top-10 hit | Long-Short |
|---|---|---|---|---|---|---|
| LogReg | 8800 | 0.5039 | 0.2518 | 0.0491 | 0.5170 | -0.0041 |
| XGBoost | 8800 | 0.4918 | 0.3303 | 0.2381 | 0.4898 | -0.0031 |
| MLP-no-graph | 8800 | 0.5153 | 0.2607 | 0.0885 | 0.5045 | 0.0074 |
| GCN | 8800 | 0.5744 | 0.2486 | 0.0171 | 0.5909 | 0.0271 |

## 2. Feature ablation (XGBoost)

| Setting | n_feat | AUC | Brier |
|---|---|---|---|
| all | 23 | 0.4918 | 0.3303 |
| no_return_vol | 13 | 0.5058 | 0.3381 |
| no_fundamental | 14 | 0.4556 | 0.2983 |
| no_momentum | 19 | 0.4893 | 0.3109 |
| no_network_stat | 23 | 0.4918 | 0.3303 |

## 3. Edge-drop ablation (GCN)

| drop_rate | AUC |
|---|---|
| 0.00 | 0.4863 |
| 0.25 | 0.4902 |
| 0.50 | 0.4932 |
| 0.75 | 0.4946 |
| 0.90 | 0.4996 |

## 4. AUC by market regime

| Model | bull | bear | sideways |
|---|---|---|---|
| LogReg | 0.5184 | 0.5108 | 0.4925 |
| XGBoost | 0.5537 | 0.4806 | 0.4917 |
| MLP-no-graph | 0.6071 | 0.4997 | 0.5228 |
| GCN | 0.5278 | 0.5636 | 0.6006 |

## 5. Per-industry highlights

Top industries by GCN AUC:
- 综合: AUC=0.7775, n=110
- 电子: AUC=0.6997, n=330
- 钢铁: AUC=0.6853, n=550
- 基础化工: AUC=0.6828, n=330
- 轻工制造: AUC=0.6597, n=220

Worst industries by GCN AUC:
- 房地产: AUC=0.3805, n=770
- 建筑装饰: AUC=0.3972, n=330
- 医药生物: AUC=0.3995, n=330
- 机械设备: AUC=0.4364, n=220
- 交通运输: AUC=0.4659, n=440