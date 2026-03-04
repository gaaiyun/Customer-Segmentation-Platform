"""
聚类分析模块 - K-Means、DBSCAN、层次聚类
用于自动客户分群和最优 K 值选择
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px


class CustomerClustering:
    """客户聚类分析器"""
    
    def __init__(self, data: pd.DataFrame, features: list = None):
        """
        初始化聚类分析器
        
        Args:
            data: 客户特征数据 DataFrame
            features: 用于聚类的特征列名列表
        """
        self.data = data.copy()
        self.features = features if features else data.columns.tolist()
        self.scaler = StandardScaler()
        self.scaled_data = None
        self.labels = None
        self.model = None
        self.method = None
        
    def prepare_data(self) -> np.ndarray:
        """
        准备并标准化数据
        
        Returns:
            标准化后的特征矩阵
        """
        feature_data = self.data[self.features].copy()
        
        # 处理缺失值
        feature_data = feature_data.fillna(feature_data.median())
        
        # 标准化
        self.scaled_data = self.scaler.fit_transform(feature_data)
        
        return self.scaled_data
    
    def find_optimal_k(self, k_range: range = range(2, 11), 
                       method: str = 'kmeans') -> Dict[str, list]:
        """
        寻找最优聚类数量
        
        Args:
            k_range: K 值搜索范围
            method: 聚类方法
            
        Returns:
            包含各种评估指标的字典
        """
        if self.scaled_data is None:
            self.prepare_data()
        
        metrics = {
            'k': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'inertia': []
        }
        
        for k in k_range:
            if method == 'kmeans':
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                raise ValueError("目前仅支持 KMeans 的最优 K 值搜索")
            
            labels = model.fit_predict(self.scaled_data)
            
            # 计算评估指标
            silhouette = silhouette_score(self.scaled_data, labels)
            davies_bouldin = davies_bouldin_score(self.scaled_data, labels)
            calinski_harabasz = calinski_harabasz_score(self.scaled_data, labels)
            inertia = model.inertia_
            
            metrics['k'].append(k)
            metrics['silhouette'].append(silhouette)
            metrics['davies_bouldin'].append(davies_bouldin)
            metrics['calinski_harabasz'].append(calinski_harabasz)
            metrics['inertia'].append(inertia)
        
        return metrics
    
    def plot_elbow_curve(self, metrics: Dict) -> go.Figure:
        """
        绘制肘部法则曲线
        
        Args:
            metrics: find_optimal_k 返回的指标字典
            
        Returns:
            Plotly 图形对象
        """
        fig = go.Figure()
        
        # 肘部曲线
        fig.add_trace(go.Scatter(
            x=metrics['k'],
            y=metrics['inertia'],
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='肘部法则 - 寻找最优 K 值',
            xaxis_title='聚类数量 (K)',
            yaxis_title='惯性 (Inertia)',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def plot_silhouette_analysis(self, metrics: Dict) -> go.Figure:
        """
        绘制轮廓系数分析图
        
        Args:
            metrics: find_optimal_k 返回的指标字典
            
        Returns:
            Plotly 图形对象
        """
        fig = go.Figure()
        
        # 轮廓系数
        fig.add_trace(go.Scatter(
            x=metrics['k'],
            y=metrics['silhouette'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='green', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title='轮廓系数分析',
            xaxis_title='聚类数量 (K)',
            yaxis_title='轮廓系数',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def kmeans_clustering(self, n_clusters: int = 4) -> np.ndarray:
        """
        执行 K-Means 聚类
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            聚类标签数组
        """
        if self.scaled_data is None:
            self.prepare_data()
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(self.scaled_data)
        self.method = 'kmeans'
        
        return self.labels
    
    def dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        执行 DBSCAN 聚类
        
        Args:
            eps: 邻域半径
            min_samples: 最小样本数
            
        Returns:
            聚类标签数组
        """
        if self.scaled_data is None:
            self.prepare_data()
        
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = self.model.fit_predict(self.scaled_data)
        self.method = 'dbscan'
        
        return self.labels
    
    def hierarchical_clustering(self, n_clusters: int = 4, 
                                linkage: str = 'ward') -> np.ndarray:
        """
        执行层次聚类
        
        Args:
            n_clusters: 聚类数量
            linkage: 连接方法 ('ward', 'complete', 'average', 'single')
            
        Returns:
            聚类标签数组
        """
        if self.scaled_data is None:
            self.prepare_data()
        
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels = self.model.fit_predict(self.scaled_data)
        self.method = 'hierarchical'
        
        return self.labels
    
    def get_cluster_statistics(self) -> pd.DataFrame:
        """
        获取各聚类的统计信息
        
        Returns:
            各聚类的统计摘要
        """
        if self.labels is None:
            raise ValueError("请先执行聚类")
        
        result = self.data.copy()
        result['cluster'] = self.labels
        
        stats = result.groupby('cluster')[self.features].agg(['mean', 'std', 'count'])
        
        return stats
    
    def get_cluster_profiles(self) -> pd.DataFrame:
        """
        获取各聚类的特征画像
        
        Returns:
            各聚类的特征均值
        """
        if self.labels is None:
            raise ValueError("请先执行聚类")
        
        result = self.data.copy()
        result['cluster'] = self.labels
        
        profiles = result.groupby('cluster')[self.features].mean()
        profiles['count'] = result.groupby('cluster').size()
        
        return profiles
    
    def plot_cluster_distribution(self, feature1: str, feature2: str = None) -> go.Figure:
        """
        绘制聚类分布图
        
        Args:
            feature1: 第一个特征 (X 轴)
            feature2: 第二个特征 (Y 轴，可选，默认使用 PCA)
            
        Returns:
            Plotly 图形对象
        """
        if self.labels is None:
            raise ValueError("请先执行聚类")
        
        if feature2 is None:
            # 使用 PCA 降维到 2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            data_2d = pca.fit_transform(self.scaled_data)
            x_col, y_col = 'PC1', 'PC2'
            plot_data = pd.DataFrame({
                'PC1': data_2d[:, 0],
                'PC2': data_2d[:, 1],
                'cluster': self.labels.astype(str)
            })
            
            fig = px.scatter(plot_data, x='PC1', y='PC2', color='cluster',
                            title='聚类分布 (PCA 降维)',
                            labels={'PC1': f'主成分 1', 'PC2': f'主成分 2'},
                            template='plotly_white')
        else:
            plot_data = pd.DataFrame({
                feature1: self.data[feature1],
                feature2: self.data[feature2],
                'cluster': self.labels.astype(str)
            })
            
            fig = px.scatter(plot_data, x=feature1, y=feature2, color='cluster',
                            title=f'聚类分布 - {feature1} vs {feature2}',
                            template='plotly_white')
        
        return fig
    
    def plot_radar_chart(self) -> go.Figure:
        """
        绘制各聚类的雷达图
        
        Returns:
            Plotly 图形对象
        """
        if self.labels is None:
            raise ValueError("请先执行聚类")
        
        profiles = self.get_cluster_profiles()
        
        fig = go.Figure()
        
        features = self.features
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        
        for cluster_id in profiles.index:
            values = profiles.loc[cluster_id, features].tolist()
            # 标准化到 0-1 范围
            values_norm = [(v - self.data[feat].min()) / (self.data[feat].max() - self.data[feat].min() + 1e-10) 
                          for v, feat in zip(values, features)]
            values_norm += values_norm[:1]
            
            fig.add_trace(go.Scatterpolar(
                r=values_norm,
                theta=[f + '<br>' for f in features] + [features[0] + '<br>'],
                fill='toself',
                name=f'Cluster {cluster_id}'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='聚类特征雷达图',
            template='plotly_white',
            height=600
        )
        
        return fig


if __name__ == "__main__":
    # 测试代码
    print("聚类分析模块测试")
    
    # 创建示例数据
    np.random.seed(42)
    n_customers = 200
    
    sample_data = pd.DataFrame({
        'recency': np.random.exponential(30, n_customers),
        'frequency': np.random.poisson(5, n_customers) + 1,
        'monetary': np.random.lognormal(3, 0.8, n_customers)
    })
    
    clusterer = CustomerClustering(sample_data)
    
    # 寻找最优 K
    print("\n寻找最优 K 值...")
    metrics = clusterer.find_optimal_k(range(2, 8))
    
    for k, sil, db in zip(metrics['k'], metrics['silhouette'], metrics['davies_bouldin']):
        print(f"K={k}: Silhouette={sil:.3f}, Davies-Bouldin={db:.3f}")
    
    # 执行聚类
    print("\n执行 K-Means 聚类 (K=4)...")
    labels = clusterer.kmeans_clustering(n_clusters=4)
    
    print("\n聚类统计:")
    print(clusterer.get_cluster_statistics())
    
    print("\n聚类画像:")
    print(clusterer.get_cluster_profiles())
