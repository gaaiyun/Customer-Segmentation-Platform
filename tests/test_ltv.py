"""
LTV 预测模块单元测试
测试覆盖率目标：>75%
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ltv_predictor import LTVPredictor


class TestLTVPredictor:
    """LTV 预测器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            'recency': np.random.exponential(30, n),
            'frequency': np.random.poisson(5, n) + 1,
            'monetary': np.random.lognormal(3, 0.8, n),
            'age': np.random.normal(35, 10, n),
            'ltv': np.random.lognormal(4, 1, n)
        })
    
    @pytest.fixture
    def predictor(self, sample_data):
        """创建预测器实例"""
        return LTVPredictor(sample_data, target_col='ltv')
    
    def test_init(self, predictor, sample_data):
        """测试初始化"""
        assert predictor.data.shape == sample_data.shape
        assert predictor.target_col == 'ltv'
        assert predictor.model is None
    
    def test_prepare_features(self, predictor):
        """测试特征准备"""
        features = predictor.prepare_features()
        
        assert 'recency' in features
        assert 'frequency' in features
        assert 'monetary' in features
        assert 'age' in features
        assert 'ltv' not in features
    
    def test_prepare_features_custom(self, sample_data):
        """测试自定义特征"""
        predictor = LTVPredictor(sample_data, target_col='ltv')
        features = predictor.prepare_features(feature_cols=['recency', 'frequency'])
        
        assert features == ['recency', 'frequency']
    
    def test_train_test_split(self, predictor):
        """测试训练测试集划分"""
        predictor.prepare_features()
        X_train, X_test, y_train, y_test = predictor.train_test_split_data(test_size=0.3)
        
        assert X_train.shape[0] == int(0.7 * len(predictor.data))
        assert X_test.shape[0] == int(0.3 * len(predictor.data))
        assert X_train.shape[1] == len(predictor.features)
    
    def test_train_linear_model(self, predictor):
        """测试线性模型训练"""
        predictor.prepare_features()
        model = predictor.train_linear_model(regularization='linear')
        
        assert predictor.model is not None
        assert 'linear' in predictor.results
        
        results = predictor.results['linear']
        assert 'test_rmse' in results
        assert 'test_r2' in results
        assert 'coefficients' in results
    
    def test_train_ridge_model(self, predictor):
        """测试 Ridge 模型训练"""
        predictor.prepare_features()
        model = predictor.train_linear_model(regularization='ridge', alpha=0.5)
        
        assert 'linear' in predictor.results
        assert predictor.results['linear']['test_rmse'] > 0
    
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
        assert predictor.results['gradient_boosting']['test_r2'] > 0
    
    def test_predict(self, predictor):
        """测试预测"""
        predictor.prepare_features()
        predictor.train_random_forest()
        
        predictions = predictor.predict()
        
        assert len(predictions) == len(predictor.data)
        assert all(predictions > 0)  # LTV 应为正数
    
    def test_predict_custom_data(self, predictor):
        """测试自定义数据预测"""
        predictor.prepare_features()
        predictor.train_random_forest()
        
        custom_data = pd.DataFrame({
            'recency': [30, 60],
            'frequency': [5, 2],
            'monetary': [500, 200],
            'age': [35, 45]
        })
        
        predictions = predictor.predict(custom_data)
        assert len(predictions) == 2
    
    def test_get_model_comparison(self, predictor):
        """测试模型比较"""
        predictor.prepare_features()
        predictor.train_linear_model()
        predictor.train_random_forest()
        
        comparison = predictor.get_model_comparison()
        
        assert isinstance(comparison, pd.DataFrame)
        assert 'Model' in comparison.columns
        assert 'Test R²' in comparison.columns
        assert len(comparison) == 2
    
    def test_calculate_ltv_distribution(self, predictor):
        """测试 LTV 分布计算"""
        predictor.prepare_features()
        predictor.train_random_forest()
        
        dist = predictor.calculate_ltv_distribution()
        
        assert 'mean' in dist
        assert 'median' in dist
        assert 'std' in dist
        assert 'min' in dist
        assert 'max' in dist
        
        assert dist['min'] <= dist['median'] <= dist['max']
    
    def test_cross_validate(self, predictor):
        """测试交叉验证"""
        predictor.prepare_features()
        cv_results = predictor.cross_validate(model_type='random_forest', cv=3)
        
        assert 'r2_mean' in cv_results
        assert 'r2_std' in cv_results
        assert 'mae_mean' in cv_results
        assert len(cv_results['r2_scores']) == 3


def test_plotting_functions():
    """测试绘图函数"""
    np.random.seed(42)
    data = pd.DataFrame({
        'recency': np.random.exponential(30, 100),
        'frequency': np.random.poisson(5, 100) + 1,
        'monetary': np.random.lognormal(3, 0.8, 100),
        'ltv': np.random.lognormal(4, 1, 100)
    })
    
    predictor = LTVPredictor(data, target_col='ltv')
    predictor.prepare_features()
    predictor.train_random_forest()
    
    # 测试特征重要性图
    fig_importance = predictor.plot_feature_importance(top_n=5)
    assert fig_importance is not None
    
    # 测试预测 vs 实际图
    fig_pred = predictor.plot_predictions_vs_actual()
    assert fig_pred is not None


def test_error_handling():
    """测试错误处理"""
    data = pd.DataFrame({
        'recency': [10, 20, 30, 40, 50],
        'ltv': [100, 200, 300, 400, 500]
    })
    
    predictor = LTVPredictor(data, target_col='ltv')
    
    # 未训练时预测应报错
    with pytest.raises(ValueError):
        predictor.predict()
    
    # 无结果时比较应报错
    with pytest.raises(ValueError):
        predictor.get_model_comparison()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
