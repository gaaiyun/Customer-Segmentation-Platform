"""
流失预测模块单元测试
测试覆盖率目标：>75%
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from churn_predictor import ChurnPredictor


class TestChurnPredictor:
    """流失预测器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据 — churn 由特征派生，让模型有信号可学。"""
        rng = np.random.default_rng(42)
        n = 400  # 200 → 400，给模型更稳定的训练样本
        recency = rng.exponential(30, n)
        frequency = rng.poisson(5, n) + 1
        monetary = rng.lognormal(3, 0.8, n)
        age = rng.normal(35, 10, n)

        # 真实 churn 信号：高 recency + 低 frequency + 低 monetary → churn 概率高
        # 加入少量随机噪声防止 perfect separation
        logit = 0.04 * recency - 0.35 * frequency - 0.08 * np.log1p(monetary) - 1.0
        prob = 1.0 / (1.0 + np.exp(-logit))
        churn_labels = (rng.random(n) < prob).astype(int)

        return pd.DataFrame({
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary,
            'age': age,
            'churn': churn_labels,
        })
    
    @pytest.fixture
    def predictor(self, sample_data):
        """创建预测器实例"""
        return ChurnPredictor(sample_data, target_col='churn')
    
    def test_init(self, predictor, sample_data):
        """测试初始化"""
        assert predictor.data.shape == sample_data.shape
        assert predictor.target_col == 'churn'
        assert set(sample_data['churn'].unique()).issubset({0, 1})
    
    def test_prepare_features(self, predictor):
        """测试特征准备"""
        features = predictor.prepare_features()
        
        assert 'recency' in features
        assert 'frequency' in features
        assert 'monetary' in features
        assert 'age' in features
        assert 'churn' not in features
    
    def test_train_test_split(self, predictor):
        """测试训练测试集划分"""
        predictor.prepare_features()
        X_train, X_test, y_train, y_test = predictor.train_test_split_data(test_size=0.25)
        
        assert X_train.shape[0] == int(0.75 * len(predictor.data))
        assert X_test.shape[0] == int(0.25 * len(predictor.data))
        
        # 检查分层抽样
        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        # 分层抽样应使流失率接近
        assert abs(train_churn_rate - test_churn_rate) < 0.1
    
    def test_train_logistic_regression(self, predictor):
        """测试逻辑回归训练"""
        predictor.prepare_features()
        model = predictor.train_logistic_regression()
        
        assert predictor.model is not None
        assert 'logistic_regression' in predictor.results
        
        results = predictor.results['logistic_regression']
        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'roc_auc' in results
        
        # ROC-AUC 应在 0.5-1 之间
        assert 0.5 <= results['roc_auc'] <= 1.0
    
    def test_train_random_forest(self, predictor):
        """测试随机森林训练"""
        predictor.prepare_features()
        model = predictor.train_random_forest(n_estimators=50)
        
        assert 'random_forest' in predictor.results
        
        results = predictor.results['random_forest']
        assert 'feature_importance' in results
        assert sum(results['feature_importance'].values()) == pytest.approx(1.0, rel=0.01)
    
    def test_train_gradient_boosting(self, predictor):
        """测试梯度提升训练"""
        predictor.prepare_features()
        model = predictor.train_gradient_boosting(n_estimators=50)
        
        assert 'gradient_boosting' in predictor.results
        assert predictor.results['gradient_boosting']['roc_auc'] > 0.5
    
    def test_predict_churn(self, predictor):
        """测试流失预测"""
        predictor.prepare_features()
        predictor.train_random_forest()
        
        predictions = predictor.predict_churn()
        
        assert len(predictions) == len(predictor.data)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_churn_probability(self, predictor):
        """测试流失概率预测"""
        predictor.prepare_features()
        predictor.train_random_forest()
        
        probabilities = predictor.predict_churn_probability()
        
        assert len(probabilities) == len(predictor.data)
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_predict_with_threshold(self, predictor):
        """测试不同阈值的预测"""
        predictor.prepare_features()
        predictor.train_random_forest()
        
        pred_low = predictor.predict_churn(threshold=0.3)
        pred_high = predictor.predict_churn(threshold=0.7)
        
        # 低阈值应预测更多流失客户
        assert pred_low.sum() >= pred_high.sum()
    
    def test_get_high_risk_customers(self, predictor):
        """测试高风险客户识别"""
        predictor.prepare_features()
        predictor.train_random_forest()
        
        high_risk = predictor.get_high_risk_customers(threshold=0.7)
        
        assert isinstance(high_risk, pd.DataFrame)
        assert 'churn_probability' in high_risk.columns
        assert all(high_risk['churn_probability'] >= 0.7)
    
    def test_get_model_comparison(self, predictor):
        """测试模型比较"""
        predictor.prepare_features()
        predictor.train_logistic_regression()
        predictor.train_random_forest()
        
        comparison = predictor.get_model_comparison()
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'Model' in comparison.columns
        assert 'ROC-AUC' in comparison.columns
        assert len(comparison) == 2
    
    def test_cross_validate(self, predictor):
        """测试交叉验证"""
        predictor.prepare_features()
        cv_results = predictor.cross_validate(model_type='random_forest', cv=3)
        
        assert 'roc_auc_mean' in cv_results
        assert 'roc_auc_std' in cv_results
        assert 'f1_mean' in cv_results
        assert len(cv_results['roc_auc_scores']) == 3


def test_plotting_functions():
    """测试绘图函数"""
    np.random.seed(42)
    n = 100
    churn_labels = np.random.choice([0, 1], n, p=[0.8, 0.2])
    
    data = pd.DataFrame({
        'recency': np.random.exponential(30, n),
        'frequency': np.random.poisson(5, n) + 1,
        'monetary': np.random.lognormal(3, 0.8, n),
        'churn': churn_labels
    })
    
    predictor = ChurnPredictor(data, target_col='churn')
    predictor.prepare_features()
    predictor.train_random_forest()
    
    # 测试 ROC 曲线
    fig_roc = predictor.plot_roc_curve()
    assert fig_roc is not None
    
    # 测试精确率 - 召回率曲线
    fig_pr = predictor.plot_precision_recall_curve()
    assert fig_pr is not None
    
    # 测试混淆矩阵
    fig_cm = predictor.plot_confusion_matrix()
    assert fig_cm is not None
    
    # 测试特征重要性
    fig_imp = predictor.plot_feature_importance(top_n=5)
    assert fig_imp is not None


def test_class_imbalance_handling():
    """类别不平衡处理：用 class_weight='balanced' 时 recall 应 > 0。"""
    rng = np.random.default_rng(42)
    n = 600  # 200 → 600，否则 5% 不平衡下少数类只有 ~10 个样本
    recency = rng.exponential(30, n)
    frequency = rng.poisson(5, n) + 1
    monetary = rng.lognormal(3, 0.8, n)
    # 真实信号 + 高基线（让 churn 真稀有 ≈ 8%）
    logit = 0.05 * recency - 0.4 * frequency - 3.0
    prob = 1.0 / (1.0 + np.exp(-logit))
    churn_labels = (rng.random(n) < prob).astype(int)

    data = pd.DataFrame({
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'churn': churn_labels,
    })

    predictor = ChurnPredictor(data, target_col='churn')
    predictor.prepare_features()
    predictor.train_random_forest(class_weight='balanced')

    results = predictor.results['random_forest']
    # class_weight='balanced' 应让少数类至少被预测出一些（recall > 0）
    assert results['recall'] > 0


def test_error_handling():
    """测试错误处理"""
    data = pd.DataFrame({
        'recency': [10, 20, 30, 40, 50],
        'churn': [0, 0, 1, 0, 0]
    })
    
    predictor = ChurnPredictor(data, target_col='churn')
    
    # 未训练时预测应报错
    with pytest.raises(ValueError):
        predictor.predict_churn()
    
    # 无结果时比较应报错
    with pytest.raises(ValueError):
        predictor.get_model_comparison()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
