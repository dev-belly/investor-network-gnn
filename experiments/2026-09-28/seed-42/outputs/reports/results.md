# Investor Network GCN — Experiment Report

## 1. Test-set metrics

| Model | n | AUC | Brier | ECE | Top-10 hit | Long-Short |
|---|---|---|---|---|---|---|
| LogReg | 7600 | 0.5085 | 0.2514 | 0.0451 | 0.5211 | -0.0035 |
| XGBoost | 7600 | 0.5013 | 0.3283 | 0.2327 | 0.4895 | -0.0002 |
| MLP-no-graph | 7600 | 0.5080 | 0.2642 | 0.1070 | 0.5092 | 0.0040 |
| GCN | 7600 | 0.5402 | 0.2495 | 0.0183 | 0.5355 | 0.0122 |

## 2. Feature ablation (XGBoost)

| Setting | n_feat | AUC | Brier |
|---|---|---|---|
| all | 23 | 0.5013 | 0.3283 |
| no_return_vol | 13 | 0.4950 | 0.3439 |
| no_fundamental | 14 | 0.4502 | 0.2994 |
| no_momentum | 19 | 0.4860 | 0.3108 |
| no_network_stat | 23 | 0.5013 | 0.3283 |

## 3. Edge-drop ablation (GCN)

| drop_rate | AUC |
|---|---|
| 0.00 | 0.5402 |
| 0.25 | 0.5426 |
| 0.50 | 0.5289 |
| 0.75 | 0.5285 |
| 0.90 | 0.5258 |

## 4. AUC by market regime

| Model | bull | bear | sideways |
|---|---|---|---|
| LogReg | 0.5168 | 0.5162 | 0.4954 |
| XGBoost | 0.5696 | 0.4871 | 0.5030 |
| MLP-no-graph | 0.6041 | 0.4899 | 0.5182 |
| GCN | 0.5413 | 0.5261 | 0.5622 |

## 5. Per-industry highlights

Top industries by GCN AUC:
- 传媒: AUC=0.7225, n=95
- 轻工制造: AUC=0.7065, n=190
- 纺织服饰: AUC=0.6717, n=95
- 钢铁: AUC=0.6698, n=475
- 综合: AUC=0.6593, n=95

Worst industries by GCN AUC:
- 机械设备: AUC=0.2615, n=190
- 建筑装饰: AUC=0.3057, n=285
- 房地产: AUC=0.3966, n=665
- 有色金属: AUC=0.4256, n=190
- 社会服务: AUC=0.4385, n=380