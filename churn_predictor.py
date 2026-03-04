"""
流失预测模块 - 客户流失风险预警
使用机器学习预测客户流失概率
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px


class ChurnPredictor:
    """客户流失预测器"""
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'churn'):
        """
        初始化流失预测器
        
        Args:
            data: 包含客户特征和流失标签的数据 DataFrame
            target_col: 流失标签列名 (0=未流失，1=流失)
        """
        self.data = data.copy()
        self.target_col = target_col
        self.features = None
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}
        self.predictions = None
        self.probabilities = None
        
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
                             random_state: int = 42,
                             stratify: bool = True) -> Tuple:
        """
        划分训练集和测试集
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            stratify: 是否分层抽样
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.features is None:
            self.prepare_features()
        
        X = self.data[self.features].fillna(self.data[self.features].median())
        y = self.data[self.target_col]
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_logistic_regression(self, C: float = 1.0, 
                                  class_weight: str = 'balanced') -> LogisticRegression:
        """
        训练逻辑回归模型
        
        Args:
            C: 正则化强度的倒数
            class_weight: 类别权重 ('balanced' 自动处理不平衡)
            
        Returns:
            训练好的模型
        """
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            random_state=42,
            max_iter=1000
        )
        
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # 评估
        self.results['logistic_regression'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'coefficients': dict(zip(self.features, self.model.coef_[0]))
        }
        
        self.predictions = y_pred
        self.probabilities = y_prob
        
        return self.model
    
    def train_random_forest(self, n_estimators: int = 100, 
                           max_depth: int = None,
                           class_weight: str = 'balanced') -> RandomForestClassifier:
        """
        训练随机森林模型
        
        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            class_weight: 类别权重
            
        Returns:
            训练好的模型
        """
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # 评估
        self.results['random_forest'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'feature_importance': dict(zip(self.features, self.model.feature_importances_))
        }
        
        self.predictions = y_pred
        self.probabilities = y_prob
        
        return self.model
    
    def train_gradient_boosting(self, n_estimators: int = 100, 
                               learning_rate: float = 0.1) -> GradientBoostingClassifier:
        """
        训练梯度提升模型
        
        Args:
            n_estimators: 树的数量
            learning_rate: 学习率
            
        Returns:
            训练好的模型
        """
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # 评估
        self.results['gradient_boosting'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'feature_importance': dict(zip(self.features, self.model.feature_importances_))
        }
        
        self.predictions = y_pred
        self.probabilities = y_prob
        
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
        
        if model_type == 'logistic':
            model = LogisticRegression(class_weight='balanced', max_iter=1000)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"未知的模型类型：{model_type}")
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
        
        return {
            'roc_auc_mean': scores.mean(),
            'roc_auc_std': scores.std(),
            'roc_auc_scores': scores.tolist(),
            'f1_mean': f1_scores.mean(),
            'f1_std': f1_scores.std()
        }
    
    def predict_churn(self, X: pd.DataFrame = None, threshold: float = 0.5) -> np.ndarray:
        """
        预测客户流失
        
        Args:
            X: 特征数据
            threshold: 分类阈值
            
        Returns:
            流失预测 (0/1)
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        if X is None:
            X = self.data[self.features]
        
        X_scaled = self.scaler.transform(X.fillna(X.median()))
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        return predictions
    
    def predict_churn_probability(self, X: pd.DataFrame = None) -> np.ndarray:
        """
        预测客户流失概率
        
        Args:
            X: 特征数据
            
        Returns:
            流失概率
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        if X is None:
            X = self.data[self.features]
        
        X_scaled = self.scaler.transform(X.fillna(X.median()))
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return probabilities
    
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
                'Accuracy': metrics.get('accuracy', np.nan),
                'Precision': metrics.get('precision', np.nan),
                'Recall': metrics.get('recall', np.nan),
                'F1 Score': metrics.get('f1', np.nan),
                'ROC-AUC': metrics.get('roc_auc', np.nan)
            })
        
        return pd.DataFrame(comparison_data).sort_values('ROC-AUC', ascending=False)
    
    def plot_roc_curve(self) -> go.Figure:
        """
        绘制 ROC 曲线
        
        Returns:
            Plotly 图形对象
        """
        if self.probabilities is None:
            raise ValueError("请先训练模型并获取预测概率")
        
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        y_true = y_test
        
        fpr, tpr, thresholds = roc_curve(y_true, self.probabilities)
        roc_auc = roc_auc_score(y_true, self.probabilities)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        
        # 对角线
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Random Classifier'
        ))
        
        fig.update_layout(
            title='ROC 曲线',
            xaxis_title='假阳性率 (FPR)',
            yaxis_title='真阳性率 (TPR)',
            template='plotly_white',
            height=500,
            width=600
        )
        
        return fig
    
    def plot_precision_recall_curve(self) -> go.Figure:
        """
        绘制精确率 - 召回率曲线
        
        Returns:
            Plotly 图形对象
        """
        if self.probabilities is None:
            raise ValueError("请先训练模型并获取预测概率")
        
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        y_true = y_test
        
        precision, recall, thresholds = precision_recall_curve(y_true, self.probabilities)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name='Precision-Recall Curve',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title='精确率 - 召回率曲线',
            xaxis_title='召回率 (Recall)',
            yaxis_title='精确率 (Precision)',
            template='plotly_white',
            height=500,
            width=600
        )
        
        return fig
    
    def plot_confusion_matrix(self) -> go.Figure:
        """
        绘制混淆矩阵
        
        Returns:
            Plotly 图形对象
        """
        if self.predictions is None:
            raise ValueError("请先训练模型并获取预测结果")
        
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        y_true = y_test
        
        cm = confusion_matrix(y_true, self.predictions)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Not Churn', 'Churn'],
            y=['Not Churn', 'Churn'],
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='混淆矩阵',
            xaxis_title='预测',
            yaxis_title='实际',
            template='plotly_white',
            height=500,
            width=500
        )
        
        # 添加数值标签
        for i in range(2):
            for j in range(2):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=str(cm[i, j]),
                    showarrow=False,
                    font=dict(size=20, color='white' if cm[i, j] > cm.max() / 2 else 'black')
                )
        
        return fig
    
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
        elif 'logistic_regression' in self.results:
            importance = {k: abs(v) for k, v in self.results['logistic_regression']['coefficients'].items()}
        
        if importance is None:
            raise ValueError("没有可用的特征重要性数据")
        
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features = [item[0] for item in sorted_importance]
        values = [item[1] for item in sorted_importance]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color='coral',
            name='Feature Importance'
        ))
        
        fig.update_layout(
            title='流失预测特征重要性 (Top {})'.format(top_n),
            xaxis_title='重要性',
            yaxis_title='特征',
            template='plotly_white',
            height=400 + len(features) * 20
        )
        
        return fig
    
    def get_high_risk_customers(self, threshold: float = 0.7, 
                               X: pd.DataFrame = None) -> pd.DataFrame:
        """
        获取高流失风险客户
        
        Args:
            threshold: 风险阈值
            X: 特征数据
            
        Returns:
            高流失风险客户 DataFrame
        """
        if X is None:
            X = self.data
        
        probabilities = self.predict_churn_probability(X)
        
        high_risk_mask = probabilities >= threshold
        
        result = X[high_risk_mask].copy()
        result['churn_probability'] = probabilities[high_risk_mask]
        result = result.sort_values('churn_probability', ascending=False)
        
        return result


if __name__ == "__main__":
    # 测试代码
    print("流失预测模块测试")
    
    # 创建示例数据
    np.random.seed(42)
    n_customers = 500
    
    # 创建不平衡的流失数据 (约 20% 流失率)
    churn_labels = np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
    
    sample_data = pd.DataFrame({
        'recency': np.random.exponential(30, n_customers),
        'frequency': np.random.poisson(5, n_customers) + 1,
        'monetary': np.random.lognormal(3, 0.8, n_customers),
        'age': np.random.normal(35, 10, n_customers),
        'churn': churn_labels
    })
    
    predictor = ChurnPredictor(sample_data, target_col='churn')
    predictor.prepare_features()
    
    print("\n训练随机森林模型...")
    predictor.train_random_forest()
    
    print("\n模型比较:")
    print(predictor.get_model_comparison())
    
    print("\n高流失风险客户数量:")
    high_risk = predictor.get_high_risk_customers(threshold=0.7)
    print(f"{len(high_risk)} 个客户 (阈值=0.7)")
