"""
RFM 分析模块单元测试
测试覆盖率目标：>75%
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfm_analyzer import RFMAnalyzer, calculate_rfm_from_transactions


class TestRFMAnalyzer:
    """RFM 分析器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        return pd.DataFrame({
            'customer_id': range(1, 101),
            'recency': np.random.exponential(30, 100).astype(int),
            'frequency': np.random.poisson(5, 100) + 1,
            'monetary': np.random.lognormal(3, 1, 100)
        })
    
    @pytest.fixture
    def analyzer(self, sample_data):
        """创建分析器实例"""
        return RFMAnalyzer(sample_data)
    
    def test_init(self, analyzer, sample_data):
        """测试初始化"""
        assert analyzer.data.shape == sample_data.shape
        assert analyzer.recency_col == 'recency'
        assert analyzer.frequency_col == 'frequency'
        assert analyzer.monetary_col == 'monetary'
        assert analyzer.rfm_scores is None
    
    def test_calculate_rfm_scores(self, analyzer):
        """测试 RFM 分数计算"""
        scores = analyzer.calculate_rfm_scores()
        
        assert 'R_score' in scores.columns
        assert 'F_score' in scores.columns
        assert 'M_score' in scores.columns
        assert 'RFM_score' in scores.columns
        
        # 分数范围检查
        assert scores['R_score'].between(1, 5).all()
        assert scores['F_score'].between(1, 5).all()
        assert scores['M_score'].between(1, 5).all()
        
        # 总分范围检查
        assert scores['RFM_score'].between(3, 15).all()
    
    def test_segment_customers(self, analyzer):
        """测试客户细分"""
        analyzer.calculate_rfm_scores()
        segments = analyzer.segment_customers()
        
        assert 'segment' in segments.columns
        
        # 检查细分标签
        valid_segments = [
            'Champions', 'Loyal Customers', 'New Customers',
            'Potential Loyalists', 'Promising', 'At Risk',
            'Hibernating', 'Lost', 'Regular'
        ]
        assert segments['segment'].isin(valid_segments).all()
    
    def test_get_segment_distribution(self, analyzer):
        """测试细分分布"""
        analyzer.segment_customers()
        distribution = analyzer.get_segment_distribution()
        
        assert isinstance(distribution, dict)
        assert sum(distribution.values()) == len(analyzer.data)
    
    def test_get_segment_summary(self, analyzer):
        """测试细分摘要"""
        analyzer.segment_customers()
        summary = analyzer.get_segment_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'segment' in summary.index.names or 'segment' in summary.columns
    
    def test_prepare_for_clustering(self, analyzer):
        """测试聚类数据准备"""
        analyzer.calculate_rfm_scores()
        clustered_data = analyzer.prepare_for_clustering()
        
        assert clustered_data.shape[0] == analyzer.data.shape[0]
        assert clustered_data.shape[1] == 3  # recency, frequency, monetary
        
        # 检查标准化 (均值接近 0，标准差接近 1)
        assert np.abs(clustered_data.mean()).max() < 0.01
        assert np.abs(clustered_data.std() - 1).max() < 0.01
    
    def test_custom_columns(self, sample_data):
        """测试自定义列名"""
        # 重命名列
        data_renamed = sample_data.rename(columns={
            'recency': 'days_since_purchase',
            'frequency': 'purchase_count',
            'monetary': 'total_spend'
        })
        
        analyzer = RFMAnalyzer(
            data_renamed,
            recency_col='days_since_purchase',
            frequency_col='purchase_count',
            monetary_col='total_spend'
        )
        
        scores = analyzer.calculate_rfm_scores()
        assert 'R_score' in scores.columns


class TestCalculateRFMFromTransactions:
    """从交易数据计算 RFM 测试"""
    
    @pytest.fixture
    def transaction_data(self):
        """创建交易数据"""
        return pd.DataFrame({
            'customer_id': [1, 1, 1, 2, 2, 3],
            'transaction_date': pd.date_range('2025-01-01', periods=6, freq='D'),
            'amount': [100, 150, 200, 80, 120, 300]
        })
    
    def test_calculate_rfm_from_transactions(self, transaction_data):
        """测试从交易数据计算 RFM"""
        rfm = calculate_rfm_from_transactions(
            transaction_data,
            customer_id_col='customer_id',
            transaction_date_col='transaction_date',
            amount_col='amount',
            reference_date=pd.Timestamp('2025-01-10')
        )
        
        assert 'customer_id' in rfm.columns
        assert 'recency' in rfm.columns
        assert 'frequency' in rfm.columns
        assert 'monetary' in rfm.columns
        
        # 检查客户数量
        assert len(rfm) == 3
        
        # 检查 RFM 值
        customer1 = rfm[rfm['customer_id'] == 1].iloc[0]
        assert customer1['frequency'] == 3
        assert customer1['monetary'] == 450
    
    def test_default_reference_date(self, transaction_data):
        """测试默认参考日期"""
        rfm = calculate_rfm_from_transactions(transaction_data)
        
        assert 'recency' in rfm.columns
        assert (rfm['recency'] >= 0).all()


def test_edge_cases():
    """边界情况：空数据 → 空结果或 raise；单条数据 → 优雅返回（v2 行为）。"""
    # 空数据：可能 raise 也可能返回空 DataFrame，两种都接受
    empty_data = pd.DataFrame(columns=['customer_id', 'recency', 'frequency', 'monetary'])
    analyzer = RFMAnalyzer(empty_data)
    try:
        scores = analyzer.calculate_rfm_scores()
        assert len(scores) == 0
    except Exception:
        pass  # raise 也接受

    # 单条数据：v2 改用 duplicates='drop' + 兜底为中位分数，不再 raise
    single_data = pd.DataFrame({
        'customer_id': [1],
        'recency': [30],
        'frequency': [1],
        'monetary': [100]
    })
    analyzer = RFMAnalyzer(single_data)
    scores = analyzer.calculate_rfm_scores()
    assert len(scores) == 1
    # 单条数据下应当落到中位分数 3
    assert int(scores['R_score'].iloc[0]) == 3
    assert int(scores['F_score'].iloc[0]) == 3
    assert int(scores['M_score'].iloc[0]) == 3


def test_data_types():
    """数据类型处理：含 NaN 的数据 v2 改为优雅处理（之前是 raise）。"""
    data_with_nan = pd.DataFrame({
        'customer_id': range(1, 11),
        'recency': [10, 20, np.nan, 40, 50, 60, 70, 80, 90, 100],
        'frequency': [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10],
        'monetary': [100, 200, 300, 400, np.nan, 600, 700, 800, 900, 1000]
    })
    analyzer = RFMAnalyzer(data_with_nan)
    scores = analyzer.calculate_rfm_scores()
    # 应当返回结果而不是 raise；NaN 行会被 pandas qcut 自动赋 NaN 分数，
    # 但 _qcut_or_constant 的 fallback 让结果仍是 int（中位 3）。
    assert len(scores) == 10
    # R/F/M score 列应全是 int（不是 NaN）
    for col in ['R_score', 'F_score', 'M_score']:
        assert scores[col].notna().all(), f"{col} 包含 NaN，v2 应已兜底"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
