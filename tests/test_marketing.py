"""
营销建议模块单元测试
测试覆盖率目标：>75%
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marketing_advisor import MarketingAdvisor


class TestMarketingAdvisor:
    """营销建议生成器测试类"""
    
    @pytest.fixture
    def advisor(self):
        """创建营销建议生成器实例"""
        return MarketingAdvisor()
    
    @pytest.fixture
    def sample_rfm_data(self):
        """创建示例 RFM 数据"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'customer_id': range(1, n + 1),
            'recency': np.random.exponential(30, n),
            'frequency': np.random.poisson(5, n) + 1,
            'monetary': np.random.lognormal(3, 0.8, n),
            'segment': np.random.choice(
                ['Champions', 'Loyal Customers', 'At Risk', 'Lost'],
                n,
                p=[0.2, 0.4, 0.3, 0.1]
            )
        })
    
    def test_init(self, advisor):
        """测试初始化"""
        assert len(advisor.segment_strategies) > 0
        assert 'Champions' in advisor.segment_strategies
        assert 'Lost' in advisor.segment_strategies
        assert len(advisor.ab_test_templates) > 0
    
    def test_generate_segment_strategy(self, advisor):
        """测试细分群体策略生成"""
        strategy = advisor.generate_segment_strategy('Champions')
        
        assert 'description' in strategy
        assert 'strategies' in strategy
        assert 'channels' in strategy
        assert 'budget_priority' in strategy
        assert 'expected_roi' in strategy
        
        assert len(strategy['strategies']) > 0
        assert len(strategy['channels']) > 0
    
    def test_generate_segment_strategy_with_data(self, advisor, sample_rfm_data):
        """测试带数据的策略生成"""
        champions_data = sample_rfm_data[sample_rfm_data['segment'] == 'Champions']
        strategy = advisor.generate_segment_strategy('Champions', champions_data)
        
        assert 'customer_count' in strategy
        assert strategy['customer_count'] == len(champions_data)
    
    def test_generate_segment_strategy_unknown(self, advisor):
        """测试未知细分群体"""
        strategy = advisor.generate_segment_strategy('Unknown Segment')
        
        assert 'error' in strategy
    
    def test_generate_campaign_plan(self, advisor):
        """测试营销活动计划生成"""
        segments = ['Champions', 'Loyal Customers', 'At Risk']
        campaign = advisor.generate_campaign_plan(
            segments=segments,
            budget=100000,
            duration_days=30
        )
        
        assert 'name' in campaign
        assert 'duration_days' in campaign
        assert 'total_budget' in campaign
        assert 'target_segments' in campaign
        assert 'timeline' in campaign
        assert 'expected_outcomes' in campaign
        assert 'recommendations' in campaign
        
        assert len(campaign['target_segments']) == 3
        assert campaign['total_budget'] == 100000
        
        # 检查时间线
        assert len(campaign['timeline']) > 0
        for phase in campaign['timeline']:
            assert 'phase' in phase
            assert 'activities' in phase
    
    def test_generate_campaign_plan_no_budget(self, advisor):
        """测试无预算的计划生成"""
        campaign = advisor.generate_campaign_plan(
            segments=['Champions'],
            budget=None
        )
        
        assert campaign['total_budget'] is None
    
    def test_generate_ab_test_plan(self, advisor):
        """测试 A/B 测试计划生成"""
        test_plan = advisor.generate_ab_test_plan(test_type='discount')
        
        assert 'test_name' in test_plan
        assert 'test_type' in test_plan
        assert 'variants' in test_plan
        assert 'metrics' in test_plan
        assert 'sample_size' in test_plan
        assert 'duration' in test_plan
        assert 'implementation_steps' in test_plan
        
        assert len(test_plan['variants']) == 2
        assert len(test_plan['metrics']) > 0
    
    def test_generate_ab_test_plan_unknown(self, advisor):
        """测试未知测试类型"""
        test_plan = advisor.generate_ab_test_plan(test_type='unknown')
        
        assert 'error' in test_plan
    
    def test_generate_personalized_recommendations(self, advisor):
        """测试个性化推荐生成"""
        # 高价值客户
        profile_vip = {
            'segment': 'Champions',
            'recency': 5,
            'frequency': 10,
            'monetary': 5000
        }
        recs_vip = advisor.generate_personalized_recommendations(profile_vip)
        assert len(recs_vip) > 0
        
        # 新客户
        profile_new = {
            'segment': 'New Customers',
            'recency': 10,
            'frequency': 1,
            'monetary': 100
        }
        recs_new = advisor.generate_personalized_recommendations(profile_new)
        assert len(recs_new) > 0
    
    def test_generate_marketing_report(self, advisor, sample_rfm_data):
        """测试营销报告生成"""
        report = advisor.generate_marketing_report(rfm_data=sample_rfm_data)
        
        assert 'report_date' in report
        assert 'executive_summary' in report
        assert 'segment_analysis' in report
        assert 'priority_actions' in report
        assert 'budget_recommendations' in report
        assert 'expected_impact' in report
        
        # 检查细分分析
        assert 'distribution' in report['segment_analysis']
        assert len(report['segment_analysis']['distribution']) > 0
        
        # 检查预算建议
        budget_rec = report['budget_recommendations']
        assert 'customer_retention' in budget_rec
        assert 'customer_acquisition' in budget_rec
    
    def test_all_segment_strategies(self, advisor):
        """测试所有细分群体策略"""
        expected_segments = [
            'Champions', 'Loyal Customers', 'New Customers',
            'Potential Loyalists', 'Promising', 'At Risk',
            'Hibernating', 'Lost', 'Regular'
        ]
        
        for segment in expected_segments:
            strategy = advisor.generate_segment_strategy(segment)
            assert 'error' not in strategy
            assert 'description' in strategy
            assert 'strategies' in strategy
    
    def test_all_ab_test_templates(self, advisor):
        """测试所有 A/B 测试模板"""
        expected_tests = ['discount', 'subject_line', 'cta', 'timing']
        
        for test_type in expected_tests:
            test_plan = advisor.generate_ab_test_plan(test_type=test_type)
            assert 'error' not in test_plan
            assert 'variants' in test_plan


def test_strategy_content():
    """测试策略内容质量"""
    advisor = MarketingAdvisor()
    
    # Champions 策略应包含 VIP 相关建议
    champions_strategy = advisor.generate_segment_strategy('Champions')
    champions_text = ' '.join(champions_strategy['strategies']).lower()
    assert 'vip' in champions_text or '专属' in champions_text
    
    # Lost 客户策略应包含召回相关建议
    lost_strategy = advisor.generate_segment_strategy('Lost')
    lost_text = ' '.join(lost_strategy['strategies']).lower()
    assert '召回' in lost_text or '优惠' in lost_text


def test_campaign_timeline():
    """测试活动时间线完整性"""
    advisor = MarketingAdvisor()
    
    campaign = advisor.generate_campaign_plan(
        segments=['Champions', 'Loyal Customers'],
        duration_days=60
    )
    
    # 检查时间线覆盖整个活动期
    phases = campaign['timeline']
    assert phases[0]['start_day'] == 1
    assert phases[-1]['end_day'] == 60
    
    # 检查各阶段不重叠
    for i in range(len(phases) - 1):
        assert phases[i]['end_day'] < phases[i + 1]['start_day']


def test_budget_allocation():
    """测试预算分配"""
    advisor = MarketingAdvisor()
    
    campaign = advisor.generate_campaign_plan(
        segments=['Champions', 'Loyal Customers', 'At Risk'],
        budget=90000
    )
    
    # 检查预算分配
    total_allocated = sum(
        seg['budget_allocation'] for seg in campaign['target_segments']
    )
    assert total_allocated == pytest.approx(90000, rel=0.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
