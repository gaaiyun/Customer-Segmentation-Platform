"""
聚类分析模块单元测试
测试覆盖率目标：>75%
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clustering import CustomerClustering


class TestCustomerClustering:
    """客户聚类分析器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        return pd.DataFrame({
            'recency': np.random.exponential(30, 100),
            'frequency': np.random.poisson(5, 100) + 1,
            'monetary': np.random.lognormal(3, 0.8, 100)
        })
    
    @pytest.fixture
    def clusterer(self, sample_data):
        """创建聚类分析器实例"""
        return CustomerClustering(sample_data)
    
    def test_init(self, clusterer, sample_data):
        """测试初始化"""
        assert clusterer.data.shape == sample_data.shape
        assert clusterer.features == ['recency', 'frequency', 'monetary']
        assert clusterer.labels is None
    
    def test_prepare_data(self, clusterer):
        """测试数据准备"""
        scaled_data = clusterer.prepare_data()
        
        assert scaled_data.shape[0] == 100
        assert scaled_data.shape[1] == 3
        
        # 检查标准化
        assert np.abs(scaled_data.mean(axis=0)).max() < 0.01
        assert np.abs(scaled_data.std(axis=0) - 1).max() < 0.01
    
    def test_find_optimal_k(self, clusterer):
        """测试最优 K 值查找"""
        clusterer.prepare_data()
        metrics = clusterer.find_optimal_k(range(2, 6))
        
        assert 'k' in metrics
        assert 'silhouette' in metrics
        assert 'davies_bouldin' in metrics
        assert 'calinski_harabasz' in metrics
        assert 'inertia' in metrics
        
        assert len(metrics['k']) == 4
        assert metrics['k'] == [2, 3, 4, 5]
        
        # 轮廓系数范围 [-1, 1]
        assert all(-1 <= s <= 1 for s in metrics['silhouette'])
    
    def test_kmeans_clustering(self, clusterer):
        """测试 K-Means 聚类"""
        clusterer.prepare_data()
        labels = clusterer.kmeans_clustering(n_clusters=4)
        
        assert len(labels) == 100
        assert clusterer.method == 'kmeans'
        assert len(set(labels)) == 4
        assert all(0 <= l < 4 for l in labels)
    
    def test_hierarchical_clustering(self, clusterer):
        """测试层次聚类"""
        clusterer.prepare_data()
        labels = clusterer.hierarchical_clustering(n_clusters=3)
        
        assert len(labels) == 100
        assert clusterer.method == 'hierarchical'
        assert len(set(labels)) == 3
    
    def test_dbscan_clustering(self, clusterer):
        """测试 DBSCAN 聚类"""
        clusterer.prepare_data()
        labels = clusterer.dbscan_clustering(eps=0.5, min_samples=5)
        
        assert len(labels) == 100
        assert clusterer.method == 'dbscan'
        # DBSCAN 可能产生噪声点 (-1)
        assert all(-1 <= l for l in labels)
    
    def test_get_cluster_statistics(self, clusterer):
        """测试聚类统计"""
        clusterer.prepare_data()
        clusterer.kmeans_clustering(n_clusters=4)
        
        stats = clusterer.get_cluster_statistics()
        
        assert isinstance(stats, pd.DataFrame)
        assert 'cluster' in stats.index.names
    
    def test_get_cluster_profiles(self, clusterer):
        """测试聚类画像"""
        clusterer.prepare_data()
        clusterer.kmeans_clustering(n_clusters=4)
        
        profiles = clusterer.get_cluster_profiles()
        
        assert isinstance(profiles, pd.DataFrame)
        assert 'count' in profiles.columns
        assert len(profiles) == 4
    
    def test_custom_features(self, sample_data):
        """测试自定义特征"""
        sample_data['age'] = np.random.normal(35, 10, 100)
        
        clusterer = CustomerClustering(sample_data, features=['recency', 'frequency'])
        assert clusterer.features == ['recency', 'frequency']
        
        scaled_data = clusterer.prepare_data()
        assert scaled_data.shape[1] == 2
    
    def test_missing_values(self):
        """测试缺失值处理"""
        data_with_nan = pd.DataFrame({
            'recency': [10, 20, np.nan, 40, 50],
            'frequency': [1, np.nan, 3, 4, 5],
            'monetary': [100, 200, 300, np.nan, 500]
        })
        
        clusterer = CustomerClustering(data_with_nan)
        scaled_data = clusterer.prepare_data()
        
        # 应该能够处理缺失值 (用中位数填充)
        assert not np.isnan(scaled_data).any()


def test_plotting_functions():
    """测试绘图函数"""
    np.random.seed(42)
    data = pd.DataFrame({
        'recency': np.random.exponential(30, 50),
        'frequency': np.random.poisson(5, 50) + 1,
        'monetary': np.random.lognormal(3, 0.8, 50)
    })
    
    clusterer = CustomerClustering(data)
    clusterer.prepare_data()
    clusterer.kmeans_clustering(n_clusters=3)
    
    # 测试肘部曲线
    metrics = clusterer.find_optimal_k(range(2, 5))
    fig_elbow = clusterer.plot_elbow_curve(metrics)
    assert fig_elbow is not None
    
    # 测试轮廓分析
    fig_silhouette = clusterer.plot_silhouette_analysis(metrics)
    assert fig_silhouette is not None
    
    # 测试聚类分布
    fig_dist = clusterer.plot_cluster_distribution('recency', 'frequency')
    assert fig_dist is not None
    
    # 测试雷达图
    fig_radar = clusterer.plot_radar_chart()
    assert fig_radar is not None


def test_error_handling():
    """测试错误处理"""
    data = pd.DataFrame({
        'recency': [10, 20, 30],
        'frequency': [1, 2, 3],
        'monetary': [100, 200, 300]
    })
    
    clusterer = CustomerClustering(data)
    
    # 未聚类时获取统计应报错
    with pytest.raises(ValueError):
        clusterer.get_cluster_statistics()
    
    # 未聚类时获取画像应报错
    with pytest.raises(ValueError):
        clusterer.get_cluster_profiles()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
