# 📊 客户细分分析平台

一个专业的客户细分与价值分析平台，用于商业决策支持。基于 RFM 模型、机器学习聚类和预测算法，提供全面的客户洞察和营销策略建议。

## ✨ 核心功能

### 1. RFM 客户价值分析
- **最近购买 (Recency)**: 分析客户最近一次购买时间
- **购买频率 (Frequency)**: 统计客户购买频次
- **购买金额 (Monetary)**: 计算客户总消费金额
- **自动评分**: 1-5 分制 RFM 评分系统
- **客户细分**: 9 类客户群体自动识别
  - Champions (冠军客户)
  - Loyal Customers (忠诚客户)
  - New Customers (新客户)
  - Potential Loyalists (潜力忠诚客户)
  - Promising (有希望客户)
  - At Risk (需关注客户)
  - Hibernating (休眠客户)
  - Lost (流失客户)
  - Regular (普通客户)

### 2. 智能聚类分析
- **K-Means 聚类**: 自动客户分群
- **层次聚类**: 树状结构分析
- **DBSCAN**: 基于密度的聚类
- **最优 K 值选择**: 肘部法则 + 轮廓系数
- **可视化**: 雷达图、分布图、特征对比

### 3. LTV 生命周期价值预测
- **多模型支持**: 线性回归、Ridge、随机森林、梯度提升
- **特征重要性分析**: 识别关键驱动因素
- **交叉验证**: 确保模型稳定性
- **分布分析**: LTV 分位数统计

### 4. 流失预警系统
- **机器学习预测**: 逻辑回归、随机森林、梯度提升
- **ROC/AUC 评估**: 模型性能全面评估
- **高风险客户识别**: 可配置风险阈值
- **混淆矩阵**: 详细分类性能分析

### 5. 智能营销建议
- **群体策略**: 针对不同细分的定制化策略
- **营销活动计划**: 完整的时间线和预算分配
- **A/B 测试方案**: 科学的测试设计
- **ROI 预测**: 预期效果评估

## 🚀 快速开始

### 环境要求
- Python 3.8+
- pip 包管理器

### 安装依赖

```bash
cd customer-segmentation-platform
pip install -r requirements.txt
```

### 启动应用

```bash
streamlit run dashboard.py
```

浏览器自动打开 `http://localhost:8501`

### 使用示例数据

平台内置示例数据，启动后选择"示例数据"即可开始分析。

### 使用自有数据

支持 CSV 和 Excel 格式，需要包含以下列:
- `customer_id`: 客户唯一标识
- `recency`: 最近购买天数
- `frequency`: 购买频率
- `monetary`: 购买金额

可选列:
- `age`: 年龄
- `gender`: 性别
- `region`: 地区
- `churn`: 流失标签 (0/1)
- `ltv`: 生命周期价值

## 📁 项目结构

```
customer-segmentation-platform/
├── dashboard.py              # 主界面 (Streamlit)
├── rfm_analyzer.py          # RFM 分析模块
├── clustering.py            # 聚类分析模块
├── ltv_predictor.py         # LTV 预测模块
├── churn_predictor.py       # 流失预测模块
├── marketing_advisor.py     # 营销建议模块
├── requirements.txt         # 依赖包列表
├── README.md               # 本文档
├── sample_customers.csv    # 示例客户数据
├── sample_transactions.csv # 示例交易数据
└── tests/                  # 单元测试目录
    ├── test_rfm.py
    ├── test_clustering.py
    ├── test_ltv.py
    ├── test_churn.py
    └── test_marketing.py
```

## 📖 使用指南

### 1. 数据加载

**方式一：使用示例数据**
- 侧边栏选择"示例数据"
- 自动加载 1000 条模拟客户数据

**方式二：上传自有数据**
- 侧边栏选择"上传文件"
- 支持 CSV/Excel 格式
- 映射必要的列名

### 2. RFM 分析

1. 选择"RFM 分析"功能
2. 查看 RFM 分数分布
3. 分析各细分群体特征
4. 使用雷达图对比群体差异

**输出指标**:
- 各群体客户数量
- 平均 R/F/M 分数
- 群体特征统计

### 3. 聚类分析

1. 选择用于聚类的特征
2. 查看肘部曲线确定最优 K
3. 选择聚类方法执行
4. 可视化聚类结果

**支持方法**:
- K-Means (推荐)
- 层次聚类
- DBSCAN

### 4. LTV 预测

1. 选择预测特征
2. 选择模型类型
3. 训练并评估模型
4. 查看预测分布

**评估指标**:
- RMSE (均方根误差)
- MAE (平均绝对误差)
- R² (决定系数)

### 5. 流失预警

1. 确认数据包含流失标签
2. 选择预测模型
3. 设置风险阈值
4. 识别高风险客户

**输出**:
- 模型性能对比
- ROC 曲线
- 高风险客户列表

### 6. 营销建议

1. 查看各群体策略
2. 生成营销活动计划
3. 设计 A/B 测试方案
4. 导出营销报告

## 🔬 技术细节

### RFM 评分算法

```python
# 分位数评分 (1-5 分)
R_score = pd.qcut(recency_rank, q=5, labels=[5,4,3,2,1])
F_score = pd.qcut(frequency_rank, q=5, labels=[1,2,3,4,5])
M_score = pd.qcut(monetary_rank, q=5, labels=[1,2,3,4,5])
RFM_total = R_score + F_score + M_score
```

### 客户细分规则

| 细分群体 | R 分数 | F 分数 | M 分数 |
|---------|-------|-------|-------|
| Champions | ≥4 | ≥4 | ≥4 |
| Loyal Customers | ≥4 | ≥3 | - |
| New Customers | ≥4 | ≤2 | - |
| At Risk | ≤2 | ≥3 | ≥3 |
| Lost | ≤2 | ≤2 | ≤2 |

### LTV 计算公式

简化版 LTV:
```
LTV = Monetary × Frequency × (12 / (Recency/30 + 1))
```

### 流失预测模型

使用平衡类别权重处理不平衡数据:
```python
RandomForestClassifier(class_weight='balanced')
```

## 🧪 测试

运行单元测试:

```bash
cd customer-segmentation-platform
pytest tests/ -v --cov=. --cov-report=html
```

查看测试覆盖率报告:
```bash
open htmlcov/index.html  # macOS/Linux
start htmlcov\index.html  # Windows
```

## 📊 可视化示例

### RFM 分布图
- 直方图展示 R/F/M 分布
- 散点图展示客户分布

### 聚类可视化
- PCA 降维散点图
- 雷达图对比群体特征
- 热力图展示聚类统计

### 模型评估
- ROC 曲线
- 混淆矩阵
- 特征重要性条形图

## 💡 最佳实践

### 1. 数据质量
- 确保数据完整性 (无缺失值)
- 处理异常值 (使用 IQR 方法)
- 定期更新数据

### 2. 模型选择
- RFM 分析：适用于大多数场景
- K-Means: 客户群体明显分离时
- 随机森林：LTV/流失预测首选

### 3. 营销策略
- 优先关注高价值客户维护
- 及时干预风险客户
- 持续 A/B 测试优化

### 4. 结果解读
- 结合业务背景理解数据
- 不要过度依赖单一指标
- 定期验证模型效果

## 🔧 高级配置

### 自定义细分规则

编辑 `rfm_analyzer.py` 中的 `assign_segment` 函数:

```python
def assign_segment(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']
    # 自定义规则
    if r >= 4 and f >= 4:
        return 'VIP Customers'
    # ...
```

### 添加新模型

在预测模块中添加新模型:

```python
def train_xgboost(self, **params):
    from xgboost import XGBClassifier
    self.model = XGBClassifier(**params)
    # ...
```

### 自定义可视化

使用 Plotly 自定义图表:

```python
import plotly.graph_objects as go
fig = go.Figure()
# 自定义配置
```

## 📝 常见问题

### Q: 数据格式要求？
A: 至少需要 customer_id, recency, frequency, monetary 四列。其他列为可选。

### Q: 如何处理缺失值？
A: 系统自动使用中位数填充数值型缺失值。建议预处理时处理缺失值。

### Q: 聚类数量如何选择？
A: 使用肘部法则和轮廓系数，通常 3-6 个聚类较为合适。

### Q: 模型准确率低怎么办？
A: 
1. 检查数据质量
2. 尝试不同模型
3. 调整特征工程
4. 增加训练数据

### Q: 如何导出结果？
A: 侧边栏选择导出格式 (CSV/Excel)，点击"导出数据"。

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📧 联系

如有问题或建议，请提交 Issue。

---

**版本**: 1.0.0  
**更新日期**: 2026-03-04  
**作者**: AI Assistant
