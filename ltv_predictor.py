"""
LTV 预测模块 - 客户生命周期价值预测
基于历史数据预测客户未来价值
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px


class LTVPredictor:
    """客户生命周期价值预测器"""
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'ltv'):
        """
        初始化 LTV 预测器
        
        Args:
            data: 包含客户特征和 LTV 的数据 DataFrame
            target_col: LTV 目标列名
        """
        self.data = data.copy()
        self.target_col = target_col
        self.features = None
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def prepare_features(self, feature_cols: list = None, 
                        exclude_cols: list = None) -> list:
        """
        准备特征列
        
        Args:
            feature_cols: 指定的特征列
            exclude_cols: 要排除的列
            
        Returns:
            特征列名列表
        """
        if feature_cols:
            self.features = feature_cols
        else:
            # 自动选择特征列
            all_cols = self.data.columns.tolist()
            exclude = [self.target_col] + (exclude_cols if exclude_cols else [])
            self.features = [col for col in all_cols if col not in exclude]
        
        return self.features
    
    def train_test_split_data(self, test_size: float = 0.2, 
                             random_state: int = 42) -> Tuple:
        """
        划分训练集和测试集
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.features is None:
            self.prepare_features()
        
        X = self.data[self.features].fillna(self.data[self.features].median())
        y = self.data[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_linear_model(self, regularization: str = 'ridge', 
                          alpha: float = 1.0) -> LinearRegression:
        """
        训练线性模型
        
        Args:
            regularization: 正则化类型 ('linear' 或 'ridge')
            alpha: Ridge 正则化强度
            
        Returns:
            训练好的模型
        """
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        
        if regularization == 'ridge':
            self.model = Ridge(alpha=alpha)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train, y_train)
        
        # 评估
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.results['linear'] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'coefficients': dict(zip(self.features, self.model.coef_))
        }
        
        return self.model
    
    def train_random_forest(self, n_estimators: int = 100, 
                           max_depth: int = None) -> RandomForestRegressor:
        """
        训练随机森林模型
        
        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            
        Returns:
            训练好的模型
        """
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.results['random_forest'] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(self.features, self.model.feature_importances_))
        }
        
        return self.model
    
    def train_gradient_boosting(self, n_estimators: int = 100, 
                               learning_rate: float = 0.1) -> GradientBoostingRegressor:
        """
        训练梯度提升模型
        
        Args:
            n_estimators: 树的数量
            learning_rate: 学习率
            
        Returns:
            训练好的模型
        """
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.results['gradient_boosting'] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(self.features, self.model.feature_importances_))
        }
        
        return self.model
    
    def cross_validate(self, model_type: str = 'random_forest', 
                      cv: int = 5) -> Dict:
        """
        交叉验证
        
        Args:
            model_type: 模型类型
            cv: 折数
            
        Returns:
            交叉验证结果
        """
        if self.features is None:
            self.prepare_features()
        
        X = self.data[self.features].fillna(self.data[self.features].median())
        y = self.data[self.target_col]
        X_scaled = self.scaler.fit_transform(X)
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
        else:
            raise ValueError(f"未知的模型类型：{model_type}")
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        mae_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
        
        return {
            'r2_mean': scores.mean(),
            'r2_std': scores.std(),
            'r2_scores': scores.tolist(),
            'mae_mean': -mae_scores.mean(),
            'mae_std': -mae_scores.std()
        }
    
    def predict(self, X: pd.DataFrame = None) -> np.ndarray:
        """
        预测 LTV
        
        Args:
            X: 特征数据
            
        Returns:
            预测的 LTV 值
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        if X is None:
            X = self.data[self.features]
        
        X_scaled = self.scaler.transform(X.fillna(X.median()))
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        获取模型比较结果
        
        Returns:
            各模型性能对比 DataFrame
        """
        if not self.results:
            raise ValueError("请先训练至少一个模型")
        
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test RMSE': metrics.get('test_rmse', np.nan),
                'Test MAE': metrics.get('test_mae', np.nan),
                'Test R²': metrics.get('test_r2', np.nan),
                'Train R²': metrics.get('train_r2', np.nan)
            })
        
        return pd.DataFrame(comparison_data).sort_values('Test R²', ascending=False)
    
    def plot_feature_importance(self, top_n: int = 10) -> go.Figure:
        """
        绘制特征重要性图
        
        Args:
            top_n: 显示前 N 个特征
            
        Returns:
            Plotly 图形对象
        """
        # 获取特征重要性
        importance = None
        
        if 'random_forest' in self.results:
            importance = self.results['random_forest']['feature_importance']
        elif 'gradient_boosting' in self.results:
            importance = self.results['gradient_boosting']['feature_importance']
        elif 'linear' in self.results:
            # 使用线性模型系数的绝对值
            importance = {k: abs(v) for k, v in self.results['linear']['coefficients'].items()}
        
        if importance is None:
            raise ValueError("没有可用的特征重要性数据")
        
        # 排序并取前 N 个
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features = [item[0] for item in sorted_importance]
        values = [item[1] for item in sorted_importance]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color='steelblue',
            name='Feature Importance'
        ))
        
        fig.update_layout(
            title='特征重要性 (Top {})'.format(top_n),
            xaxis_title='重要性',
            yaxis_title='特征',
            template='plotly_white',
            height=400 + len(features) * 20
        )
        
        return fig
    
    def plot_predictions_vs_actual(self) -> go.Figure:
        """
        绘制预测值 vs 实际值散点图
        
        Returns:
            Plotly 图形对象
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        y_pred = self.model.predict(X_test)
        
        fig = go.Figure()
        
        # 散点图
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            marker=dict(size=8, opacity=0.6),
            name='Predictions'
        ))
        
        # 理想线
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        
        fig.update_layout(
            title='预测值 vs 实际值',
            xaxis_title='实际 LTV',
            yaxis_title='预测 LTV',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def calculate_ltv_distribution(self, predictions: np.ndarray = None) -> Dict:
        """
        计算 LTV 分布统计
        
        Args:
            predictions: 预测的 LTV 值
            
        Returns:
            LTV 分布统计字典
        """
        if predictions is None:
            predictions = self.predict()
        
        return {
            'mean': float(np.mean(predictions)),
            'median': float(np.median(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75)),
            'q90': float(np.percentile(predictions, 90))
        }


if __name__ == "__main__":
    # 测试代码
    print("LTV 预测模块测试")
    
    # 创建示例数据
    np.random.seed(42)
    n_customers = 500
    
    sample_data = pd.DataFrame({
        'recency': np.random.exponential(30, n_customers),
        'frequency': np.random.poisson(5, n_customers) + 1,
        'monetary': np.random.lognormal(3, 0.8, n_customers),
        'age': np.random.normal(35, 10, n_customers),
        'ltv': np.random.lognormal(4, 1, n_customers)
    })
    
    predictor = LTVPredictor(sample_data, target_col='ltv')
    predictor.prepare_features()
    
    print("\n训练随机森林模型...")
    predictor.train_random_forest()
    
    print("\n模型比较:")
    print(predictor.get_model_comparison())
    
    print("\nLTV 分布:")
    dist = predictor.calculate_ltv_distribution()
    for k, v in dist.items():
        print(f"{k}: {v:.2f}")
