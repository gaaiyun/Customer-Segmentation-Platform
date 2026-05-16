# Customer Segmentation Platform

> RFM + 聚类 + ML 是入门，**BG/NBD + Gamma-Gamma + Cohort + 模型解释**才是 v2。

电商 / SaaS / 订阅业务的客户分析栈：从「最近一次买了什么」到「未来一年还会买多少、值多少钱、为什么可能流失」一站式打通。

## v2 新增

| 模块 | 解决什么问题 | 文献 |
|---|---|---|
| `bgnbd.py` | 概率 CLV：从交易记录估出"未来 N 期会买几次 × 每次花多少" | [Fader, Hardie, Lee (2005)](http://brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) / [Fader, Hardie (2013)](http://brucehardie.com/notes/025/gamma_gamma.pdf) |
| `cohort.py` | 客户队列分析（按首购月分组，看后续留存与 ARPU） | 电商基本盘报表 |
| `churn_explain.py` | 模型解释：permutation importance + 单客户特征贡献（可选 SHAP） | scikit-learn `inspection` |
| `.gitignore` | v1 没有，导致 `__pycache__/` / `.coverage` 一直在 tracking | — |

外加修复：

- ✅ `rfm_analyzer.py:49` 在样本量 < 5 或同值数据下 `pd.qcut` 崩溃的 bug
- ✅ 清理 `__pycache__/` 和 `.coverage` 入库

## 快速开始

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

### 用 BG/NBD 算 CLV

```python
import pandas as pd
from bgnbd import BGNBDModel, GammaGammaModel, rfm_to_bgnbd_input, clv

# 原始交易表：[customer_id, date, amount]
txns = pd.read_csv("transactions.csv")

# 转成 BG/NBD 需要的 [frequency, recency, T, monetary_value]
rfm_input = rfm_to_bgnbd_input(txns)

# 仅对有重复交易的客户 fit Gamma-Gamma
repeat = rfm_input[rfm_input["frequency"] >= 1]

bg = BGNBDModel().fit(rfm_input)
gg = GammaGammaModel().fit(repeat)

# 未来 365 天每个客户的预期 CLV
rfm_input["clv_365d"] = clv(bg, gg, rfm_input, horizon=365)

# 未来 30 天预期交易次数
rfm_input["expected_purchases_30d"] = bg.predict_purchases(t=30, df=rfm_input)

# P(客户当前仍未流失)
rfm_input["p_alive"] = bg.predict_alive(rfm_input)
```

### Cohort 留存

```python
from cohort import build_cohort_report, average_retention_curve

report = build_cohort_report(txns, period="M")
print(report.retention.iloc[:5, :6])      # 留存矩阵
print(report.arpu.iloc[:5, :6])           # ARPU 矩阵
print(average_retention_curve(report))    # 跨 cohort 平均留存曲线
```

### 流失模型解释

```python
from churn_explain import permutation_importance_table, top_features_for_customer

# 已经 fit 好的 sklearn 模型 + 测试集
importance = permutation_importance_table(churn_model, X_test, y_test, n_repeats=10)
print(importance)

# 单个客户为什么被预测为流失？
why = top_features_for_customer(
    churn_model, X_test.iloc[0], feature_names=X_test.columns,
    feature_means=X_test.mean(), top_k=5,
)
for c in why:
    print(f"{c.feature}: 该客户值={c.value:.2f}, 贡献={c.contribution:+.3f}")
```

需要更精细的 SHAP 解释？

```bash
pip install shap
```

```python
from churn_explain import compute_shap_values
shap_data = compute_shap_values(churn_model, X_test)
# shap_data["shap_values"] 是 (n_samples, n_features) 矩阵
```

## 模块布局

```
.
├─ rfm_analyzer.py        # v1：RFM 分数 + 9 类客户细分（v2 修了 qcut 边界 bug）
├─ clustering.py          # v1：K-Means / DBSCAN / 层次聚类
├─ ltv_predictor.py       # v1：ML 模型（LR / RF / GBDT）预测 LTV
├─ churn_predictor.py     # v1：流失模型（同上三种）
├─ marketing_advisor.py   # v1：基于细分的营销建议生成器
│
├─ bgnbd.py               # v2：概率 CLV（BG/NBD + Gamma-Gamma）
├─ cohort.py              # v2：cohort × period_offset 留存与 ARPU
├─ churn_explain.py       # v2：permutation importance + 单客户解释
│
├─ dashboard.py           # Streamlit UI
└─ tests/                 # 88 个测试通过（含 25 个 v2 新测试）
```

## 何时用 BG/NBD vs LTVPredictor (v1 ML)

| 维度 | BG/NBD + Gamma-Gamma | v1 LTVPredictor (LR/RF/GBDT) |
|---|---|---|
| 输入 | 交易日志（购买频率 + 金额） | 任意特征（demo / behavior / RFM） |
| 数据需求 | 少量重复购买就够 | 大样本 |
| 假设 | 客户行为符合概率模型 | 端到端拟合 |
| 解释性 | 强（每个参数有意义） | 弱（黑盒） |
| 适合场景 | 电商 / 订阅 / 任何 transactional 业务 | 多源特征丰富的场景 |
| 风险 | 假设客户购买间隔独立、金额与频率独立 | 容易过拟合短期波动 |

实务建议：**两种都跑，对比结果**。BG/NBD 的预测如果跟 ML 模型一致 → 高置信；如果差很多 → 看是哪个模型在哪类客户上偏，通常是少数极端客户被某一方误判。

## 路线图

详见 [ROADMAP.md](ROADMAP.md)。重点：
- PDF / Excel / Word 报告导出
- Retention curve 拟合（Geometric 衰减模型）→ "新客户预计 X 个月后留存 Y%"
- Lookalike scoring（基于客户特征找相似潜客）

## 测试

```bash
python -m pytest tests/ -o addopts=""
```

> 当前 88/93 通过。失败的 5 个集中在 v1 的 churn/ltv 模型测试，是 sklearn 模型在小样本测试数据上 metric 不稳，与 v2 改动无关。

## 许可

MIT
