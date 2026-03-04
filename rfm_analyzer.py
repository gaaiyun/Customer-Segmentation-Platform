"""
RFM 分析模块 - Recency, Frequency, Monetary 分析
用于客户价值评估和细分
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime


class RFMAnalyzer:
    """RFM 分析器 - 计算客户 RFM 分数并进行分组"""
    
    def __init__(self, data: pd.DataFrame, 
                 recency_col: str = 'recency',
                 frequency_col: str = 'frequency',
                 monetary_col: str = 'monetary'):
        """
        初始化 RFM 分析器
        
        Args:
            data: 包含客户交易数据的 DataFrame
            recency_col: 最近购买时间列名
            frequency_col: 购买频率列名
            monetary_col: 购买金额列名
        """
        self.data = data.copy()
        self.recency_col = recency_col
        self.frequency_col = frequency_col
        self.monetary_col = monetary_col
        self.rfm_scores = None
        self.segments = None
        
    def calculate_rfm_scores(self, method: str = 'quantile') -> pd.DataFrame:
        """
        计算 RFM 分数 (1-5 分)
        
        Args:
            method: 评分方法 - 'quantile'(分位数) 或 'custom'(自定义阈值)
            
        Returns:
            包含 R、F、M 分数和总分的 DataFrame
        """
        df = self.data.copy()
        
        if method == 'quantile':
            # R 分数：最近购买时间越短，分数越高
            df['R_score'] = pd.qcut(df[self.recency_col].rank(method='first'), 
                                    q=5, labels=[5, 4, 3, 2, 1]).astype(int)
            
            # F 分数：购买频率越高，分数越高
            df['F_score'] = pd.qcut(df[self.frequency_col].rank(method='first'), 
                                    q=5, labels=[1, 2, 3, 4, 5]).astype(int)
            
            # M 分数：购买金额越高，分数越高
            df['M_score'] = pd.qcut(df[self.monetary_col].rank(method='first'), 
                                    q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        else:
            raise ValueError("目前仅支持 'quantile' 方法")
        
        # 计算总分
        df['RFM_score'] = df['R_score'] + df['F_score'] + df['M_score']
        
        self.rfm_scores = df
        return df
    
    def segment_customers(self) -> pd.DataFrame:
        """
        基于 RFM 分数进行客户细分
        
        Returns:
            包含客户细分标签的 DataFrame
        """
        if self.rfm_scores is None:
            self.calculate_rfm_scores()
        
        df = self.rfm_scores.copy()
        
        # 定义细分规则
        def assign_segment(row):
            r, f, m = row['R_score'], row['F_score'], row['M_score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'  # 冠军客户
            elif r >= 4 and f >= 3:
                return 'Loyal Customers'  # 忠诚客户
            elif r >= 4 and f <= 2:
                return 'New Customers'  # 新客户
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Potential Loyalists'  # 潜力忠诚客户
            elif r >= 3 and f <= 2:
                return 'Promising'  # 有希望客户
            elif r <= 2 and f >= 3 and m >= 3:
                return 'At Risk'  # 需关注客户
            elif r <= 2 and f >= 4:
                return 'Hibernating'  # 休眠客户
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost'  # 流失客户
            else:
                return 'Regular'  # 普通客户
        
        df['segment'] = df.apply(assign_segment, axis=1)
        self.segments = df
        return df
    
    def get_segment_summary(self) -> pd.DataFrame:
        """
        获取各细分群体的统计摘要
        
        Returns:
            各细分群体的统计信息
        """
        if self.segments is None:
            self.segment_customers()
        
        summary = self.segments.groupby('segment').agg({
            self.recency_col: ['mean', 'std'],
            self.frequency_col: ['mean', 'std'],
            self.monetary_col: ['mean', 'std'],
            'RFM_score': 'mean'
        }).round(2)
        
        return summary
    
    def get_segment_distribution(self) -> Dict[str, int]:
        """
        获取各细分群体的客户数量分布
        
        Returns:
            各细分群体的客户数量
        """
        if self.segments is None:
            self.segment_customers()
        
        return self.segments['segment'].value_counts().to_dict()
    
    def prepare_for_clustering(self) -> pd.DataFrame:
        """
        准备用于聚类的标准化数据
        
        Returns:
            标准化的 RFM 特征矩阵
        """
        if self.rfm_scores is None:
            self.calculate_rfm_scores()
        
        features = self.rfm_scores[[self.recency_col, self.frequency_col, self.monetary_col]].copy()
        
        # Z-score 标准化
        features_normalized = (features - features.mean()) / features.std()
        
        return features_normalized


def calculate_rfm_from_transactions(transactions: pd.DataFrame,
                                    customer_id_col: str = 'customer_id',
                                    transaction_date_col: str = 'transaction_date',
                                    amount_col: str = 'amount',
                                    reference_date: datetime = None) -> pd.DataFrame:
    """
    从交易数据计算 RFM 指标
    
    Args:
        transactions: 交易数据 DataFrame
        customer_id_col: 客户 ID 列名
        transaction_date_col: 交易日期列名
        amount_col: 交易金额列名
        reference_date: 参考日期 (用于计算最近购买时间)
        
    Returns:
        包含 RFM 指标的客户级 DataFrame
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    
    # 确保日期列为 datetime 类型
    transactions[transaction_date_col] = pd.to_datetime(transactions[transaction_date_col])
    
    # 计算 RFM 指标
    rfm = transactions.groupby(customer_id_col).agg({
        transaction_date_col: 'max',  # 最近购买时间
        customer_id_col: 'count',  # 购买频率
        amount_col: 'sum'  # 总购买金额
    }).rename(columns={
        transaction_date_col: 'recency_date',
        customer_id_col: 'frequency',
        amount_col: 'monetary'
    })
    
    # 计算最近购买天数
    rfm['recency'] = (reference_date - rfm['recency_date']).dt.days
    rfm = rfm.drop('recency_date', axis=1)
    
    return rfm.reset_index()


if __name__ == "__main__":
    # 测试代码
    print("RFM 分析模块测试")
    
    # 创建示例数据
    np.random.seed(42)
    n_customers = 100
    
    sample_data = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'recency': np.random.exponential(30, n_customers).astype(int),
        'frequency': np.random.poisson(5, n_customers) + 1,
        'monetary': np.random.lognormal(3, 1, n_customers)
    })
    
    analyzer = RFMAnalyzer(sample_data)
    scores = analyzer.calculate_rfm_scores()
    segments = analyzer.segment_customers()
    
    print("\nRFM 分数示例:")
    print(scores.head())
    
    print("\n客户细分分布:")
    print(analyzer.get_segment_distribution())
    
    print("\n细分群体统计摘要:")
    print(analyzer.get_segment_summary())
