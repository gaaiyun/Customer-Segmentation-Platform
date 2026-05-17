# Roadmap

## v2（本次提交，已完成）

- ✅ `bgnbd.py` — BG/NBD + Gamma-Gamma 概率 CLV 模型
- ✅ `cohort.py` — Cohort × period_offset 留存与 ARPU 矩阵
- ✅ `churn_explain.py` — Permutation importance + 单客户特征贡献 + 可选 SHAP
- ✅ 修 `rfm_analyzer.py:49` 的 qcut 边界 bug（< 5 行 / 同值数据下崩溃）
- ✅ 25 个新测试（BG/NBD 11 / Cohort 7 / Churn-Explain 7）
- ✅ 加 `.gitignore`，清 `__pycache__/` `.coverage` 入库

## v3 计划

### CLV 增强
- [ ] **Pareto/NBD**（BG/NBD 的更精确版本，允许中途流失而非仅在交易点流失）
- [ ] CLV 的不确定性区间（用 bootstrap 或 Hessian）
- [ ] CLV-segmentation：用 CLV 而非 RFM 做细分

### 留存 / 行为
- [ ] Geometric / Shifted-Beta-Geometric 留存曲线拟合
- [ ] Multi-state customer journey 模型（活跃 → 沉睡 → 流失）

### Churn 模型增强
- [ ] 类别不平衡处理（SMOTE / class_weight 默认开）
- [ ] 时间序列特征（滚动 7d / 30d 行为差）
- [ ] Survival analysis：what's the expected time to churn

### 工程
- [ ] CLI 入口（无需 Streamlit 跑 RFM/CLV/cohort）
- [ ] PDF / Excel 报告导出
- [ ] CI（GitHub Actions 跑 pytest）

### v1 测试遗留
- [ ] 修 5 个原有 v1 测试（churn 模型 recall=0、ltv GBDT R² < 阈值），需调整测试数据生成
- [ ] `__init__.py` 让 tests/ 成为包，避免每个 test_*.py 重复 sys.path 操作

## 不会做的

- 不会改 v1 已有的 5 个模块的接口（`RFMAnalyzer` / `Clustering` / `LTVPredictor`
  / `ChurnPredictor` / `MarketingAdvisor`），只在它们旁边加新东西
- 不会引入需要 GPU 的深度学习 baseline（CLV 任务上 ROI 太低）
