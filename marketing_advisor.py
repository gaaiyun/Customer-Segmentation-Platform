"""
营销建议模块 - 基于客户细分的精准营销策略
生成个性化的营销建议和活动方案
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class MarketingAdvisor:
    """营销建议生成器"""
    
    def __init__(self):
        """初始化营销建议生成器"""
        
        # 定义各细分群体的特征和策略
        self.segment_strategies = {
            'Champions': {
                'description': '冠军客户 - 最近购买、高频、高消费',
                'characteristics': ['高忠诚度', '高价值', '活跃'],
                'strategies': [
                    '提供 VIP 专属优惠和早期访问权',
                    '邀请参与新品试用和反馈',
                    '推荐高价值产品和增值服务',
                    '建立品牌大使计划',
                    '提供个性化推荐和专属客服'
                ],
                'channels': ['Email', 'APP 推送', '专属客服'],
                'budget_priority': '高',
                'expected_roi': '非常高'
            },
            'Loyal Customers': {
                'description': '忠诚客户 - 经常购买，消费稳定',
                'characteristics': ['稳定消费', '品牌忠诚', '中等频率'],
                'strategies': [
                    '提供忠诚度奖励和积分计划',
                    '推荐相关产品和交叉销售',
                    '发送个性化优惠券',
                    '邀请参加会员专属活动',
                    '提供订阅服务优惠'
                ],
                'channels': ['Email', '短信', 'APP 推送'],
                'budget_priority': '中高',
                'expected_roi': '高'
            },
            'New Customers': {
                'description': '新客户 - 最近购买但频率低',
                'characteristics': ['新注册', '首次购买', '潜力大'],
                'strategies': [
                    '发送欢迎礼包和首单优惠',
                    '提供新手引导和使用教程',
                    '推荐热门产品和畅销品',
                    '设置新客专属任务奖励',
                    '建立新客培育流程'
                ],
                'channels': ['Email', '短信', 'APP 推送', '社交媒体'],
                'budget_priority': '高',
                'expected_roi': '中高'
            },
            'Potential Loyalists': {
                'description': '潜力忠诚客户 - 有成为忠诚客户的潜力',
                'characteristics': ['中等活跃', '有潜力', '需要培养'],
                'strategies': [
                    '提供会员升级激励',
                    '推荐高评价产品',
                    '发送限时优惠活动',
                    '建立互动和参与机制',
                    '提供个性化内容推荐'
                ],
                'channels': ['Email', 'APP 推送', '社交媒体'],
                'budget_priority': '中',
                'expected_roi': '中高'
            },
            'Promising': {
                'description': '有希望客户 - 新客户但表现良好',
                'characteristics': ['新客', '有购买意愿', '需要引导'],
                'strategies': [
                    '提供第二次购买激励',
                    '发送产品使用指南',
                    '推荐相关配件和补充品',
                    '建立新客培育邮件序列',
                    '提供限时折扣券'
                ],
                'channels': ['Email', '短信', 'APP 推送'],
                'budget_priority': '中',
                'expected_roi': '中'
            },
            'At Risk': {
                'description': '需关注客户 - 曾经活跃但最近不活跃',
                'characteristics': ['曾经高价值', '活跃度下降', '流失风险'],
                'strategies': [
                    '发送唤醒优惠和召回活动',
                    '提供特别折扣和限时优惠',
                    '询问反馈和改进建议',
                    '推送新品和热门产品',
                    '建立流失预警和干预机制'
                ],
                'channels': ['Email', '短信', '电话'],
                'budget_priority': '高',
                'expected_roi': '中'
            },
            'Hibernating': {
                'description': '休眠客户 - 长时间未购买',
                'characteristics': ['长时间未活跃', '曾经购买', '需要唤醒'],
                'strategies': [
                    '发送强力召回优惠',
                    '提供大幅折扣和买赠活动',
                    '推送品牌动态和新品',
                    '建立休眠客户唤醒计划',
                    '考虑再营销活动'
                ],
                'channels': ['Email', '短信', '再营销广告'],
                'budget_priority': '低中',
                'expected_roi': '低中'
            },
            'Lost': {
                'description': '流失客户 - 长时间未购买且价值低',
                'characteristics': ['长期未活跃', '低价值', '高流失风险'],
                'strategies': [
                    '发送最后召回优惠',
                    '提供超值优惠和清仓活动',
                    '收集流失原因反馈',
                    '降低联系频率避免打扰',
                    '考虑从活跃名单中移除'
                ],
                'channels': ['Email', '再营销广告'],
                'budget_priority': '低',
                'expected_roi': '低'
            },
            'Regular': {
                'description': '普通客户 - 表现中等',
                'characteristics': ['一般活跃', '中等价值', '大多数客户'],
                'strategies': [
                    '提供常规促销和折扣',
                    '推荐个性化产品',
                    '发送定期通讯和更新',
                    '建立客户参与计划',
                    '提供积分和奖励'
                ],
                'channels': ['Email', 'APP 推送', '社交媒体'],
                'budget_priority': '中',
                'expected_roi': '中'
            }
        }
        
        # 定义 A/B 测试建议
        self.ab_test_templates = {
            'discount': {
                'name': '折扣力度测试',
                'variants': ['8 折优惠', '满 100 减 30'],
                'metrics': ['转化率', '客单价', 'ROI'],
                'sample_size': '每组至少 1000 人',
                'duration': '7-14 天'
            },
            'subject_line': {
                'name': '邮件主题行测试',
                'variants': ['直接型 vs 好奇型', '长标题 vs 短标题'],
                'metrics': ['打开率', '点击率', '转化率'],
                'sample_size': '每组至少 500 人',
                'duration': '3-7 天'
            },
            'cta': {
                'name': '行动号召测试',
                'variants': ['立即购买 vs 了解详情', '不同按钮颜色'],
                'metrics': ['点击率', '转化率'],
                'sample_size': '每组至少 300 人',
                'duration': '5-10 天'
            },
            'timing': {
                'name': '发送时间测试',
                'variants': ['早上 9 点 vs 晚上 8 点', '工作日 vs 周末'],
                'metrics': ['打开率', '点击率', '转化率'],
                'sample_size': '每组至少 500 人',
                'duration': '7-14 天'
            }
        }
    
    def generate_segment_strategy(self, segment: str, 
                                  customer_data: pd.DataFrame = None) -> Dict:
        """
        生成特定细分群体的营销策略
        
        Args:
            segment: 客户细分名称
            customer_data: 该细分群体的客户数据
            
        Returns:
            营销策略字典
        """
        if segment not in self.segment_strategies:
            return {'error': f'未知的细分群体：{segment}'}
        
        strategy = self.segment_strategies[segment].copy()
        
        # 如果有客户数据，添加统计信息
        if customer_data is not None:
            strategy['customer_count'] = len(customer_data)
            
            # 计算平均价值
            value_cols = ['monetary', 'ltv', 'total_spend']
            for col in value_cols:
                if col in customer_data.columns:
                    strategy[f'avg_{col}'] = customer_data[col].mean()
                    break
            
            # 计算平均活跃度
            activity_cols = ['frequency', 'days_since_last_purchase', 'recency']
            for col in activity_cols:
                if col in customer_data.columns:
                    strategy[f'avg_{col}'] = customer_data[col].mean()
                    break
        
        return strategy
    
    def generate_campaign_plan(self, segments: List[str], 
                              budget: float = None,
                              duration_days: int = 30) -> Dict:
        """
        生成营销活动计划
        
        Args:
            segments: 目标细分群体列表
            budget: 总预算 (可选)
            duration_days: 活动持续天数
            
        Returns:
            营销活动计划
        """
        campaign = {
            'name': f'精准营销活动 - {datetime.now().strftime("%Y-%m-%d")}',
            'duration_days': duration_days,
            'total_budget': budget,
            'target_segments': [],
            'timeline': [],
            'expected_outcomes': {},
            'recommendations': []
        }
        
        # 为每个细分群体制定策略
        budget_per_segment = budget / len(segments) if budget else None
        
        for segment in segments:
            if segment in self.segment_strategies:
                strategy = self.segment_strategies[segment]
                
                segment_plan = {
                    'segment': segment,
                    'description': strategy['description'],
                    'strategies': strategy['strategies'],
                    'channels': strategy['channels'],
                    'budget_allocation': budget_per_segment,
                    'priority': strategy['budget_priority']
                }
                
                campaign['target_segments'].append(segment_plan)
        
        # 生成时间线
        phases = ['准备期', '启动期', '执行期', '收尾期']
        phase_duration = duration_days // len(phases)
        
        for i, phase in enumerate(phases):
            campaign['timeline'].append({
                'phase': phase,
                'start_day': i * phase_duration + 1,
                'end_day': (i + 1) * phase_duration,
                'activities': self._get_phase_activities(phase, segments)
            })
        
        # 预期结果
        campaign['expected_outcomes'] = {
            'conversion_lift': '10-25%',
            'revenue_increase': '15-30%',
            'customer_retention': '5-15% 提升',
            'roi': '3:1 - 5:1'
        }
        
        # 总体建议
        campaign['recommendations'] = [
            '优先关注高价值细分群体 (Champions, Loyal Customers)',
            '为不同群体定制个性化内容',
            '设置明确的转化目标和 KPI',
            '进行 A/B 测试优化效果',
            '实时监控并调整策略'
        ]
        
        return campaign
    
    def _get_phase_activities(self, phase: str, segments: List[str]) -> List[str]:
        """获取各阶段的活动列表"""
        
        activities = {
            '准备期': [
                '确定目标受众和细分群体',
                '设计创意素材和文案',
                '设置追踪和分析系统',
                '准备 A/B 测试方案'
            ],
            '启动期': [
                '发送预热邮件/通知',
                '启动社交媒体宣传',
                '开始小范围测试',
                '监控初始反馈'
            ],
            '执行期': [
                '全面推广活动',
                '持续优化投放策略',
                '进行 A/B 测试',
                '收集用户反馈'
            ],
            '收尾期': [
                '活动效果分析',
                '生成总结报告',
                '规划后续跟进',
                '归档学习经验'
            ]
        }
        
        return activities.get(phase, [])
    
    def generate_ab_test_plan(self, test_type: str = 'discount',
                             segment: str = None) -> Dict:
        """
        生成 A/B 测试计划
        
        Args:
            test_type: 测试类型
            segment: 目标细分群体
            
        Returns:
            A/B 测试计划
        """
        if test_type not in self.ab_test_templates:
            return {'error': f'未知的测试类型：{test_type}'}
        
        template = self.ab_test_templates[test_type]
        
        test_plan = {
            'test_name': f'{template["name"]} - {datetime.now().strftime("%Y-%m-%d")}',
            'test_type': test_type,
            'target_segment': segment,
            'hypothesis': f'变体 B 比变体 A 在关键指标上有显著提升',
            'variants': template['variants'],
            'metrics': template['metrics'],
            'sample_size': template['sample_size'],
            'duration': template['duration'],
            'success_criteria': '统计显著性 p < 0.05，提升幅度 > 5%',
            'implementation_steps': [
                '确定测试目标和假设',
                '设计测试变体',
                '随机分配测试群体',
                '设置数据追踪',
                '运行测试并监控',
                '分析结果并得出结论',
                '实施获胜方案'
            ]
        }
        
        return test_plan
    
    def generate_personalized_recommendations(self, customer_profile: Dict) -> List[str]:
        """
        生成个性化推荐
        
        Args:
            customer_profile: 客户画像字典
            
        Returns:
            推荐列表
        """
        recommendations = []
        
        segment = customer_profile.get('segment', 'Regular')
        recency = customer_profile.get('recency', 30)
        frequency = customer_profile.get('frequency', 1)
        monetary = customer_profile.get('monetary', 100)
        
        # 基于 RFM 的推荐
        if recency < 7 and frequency > 5:
            recommendations.append('推荐 VIP 专属产品和早期访问权')
        elif recency < 30 and monetary > 500:
            recommendations.append('推荐高价值产品和增值服务')
        elif recency > 60 and frequency > 3:
            recommendations.append('发送召回优惠和限时折扣')
        elif frequency == 1:
            recommendations.append('提供新客引导和第二次购买激励')
        
        # 基于细分的推荐
        if segment in self.segment_strategies:
            strategies = self.segment_strategies[segment]['strategies']
            recommendations.extend(strategies[:2])
        
        return recommendations
    
    def generate_marketing_report(self, rfm_data: pd.DataFrame = None,
                                 clustering_data: pd.DataFrame = None,
                                 churn_predictions: pd.DataFrame = None) -> Dict:
        """
        生成综合营销报告
        
        Args:
            rfm_data: RFM 分析数据
            clustering_data: 聚类分析数据
            churn_predictions: 流失预测数据
            
        Returns:
            综合营销报告
        """
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d'),
            'executive_summary': '',
            'segment_analysis': {},
            'priority_actions': [],
            'budget_recommendations': {},
            'expected_impact': {}
        }
        
        # 细分群体分析
        if rfm_data is not None and 'segment' in rfm_data.columns:
            segment_counts = rfm_data['segment'].value_counts().to_dict()
            report['segment_analysis']['distribution'] = segment_counts
            
            # 计算各群体的平均价值
            if 'monetary' in rfm_data.columns:
                segment_value = rfm_data.groupby('segment')['monetary'].mean().to_dict()
                report['segment_analysis']['avg_value'] = segment_value
            
            # 生成执行摘要
            total_customers = len(rfm_data)
            high_value_segments = ['Champions', 'Loyal Customers']
            high_value_count = sum(segment_counts.get(seg, 0) for seg in high_value_segments)
            
            report['executive_summary'] = (
                f"总客户数：{total_customers:,}。"
                f"高价值客户占比：{high_value_count/total_customers*100:.1f}%。"
                f"建议重点关注 {', '.join(high_value_segments)} 群体的维护和增值。"
            )
        
        # 优先行动
        report['priority_actions'] = [
            '立即启动 Champions 客户维护计划',
            '针对 At Risk 客户实施召回活动',
            '为 New Customers 设计培育流程',
            '优化 Loyal Customers 的忠诚度计划'
        ]
        
        # 预算建议
        report['budget_recommendations'] = {
            'customer_retention': '40%',
            'customer_acquisition': '30%',
            'reactivation': '20%',
            'testing_optimization': '10%'
        }
        
        # 预期影响
        report['expected_impact'] = {
            'revenue_lift': '15-25%',
            'retention_improvement': '10-20%',
            'customer_lifetime_value': '提升 20-30%',
            'marketing_roi': '提升 2-3 倍'
        }
        
        return report


if __name__ == "__main__":
    # 测试代码
    print("营销建议模块测试")
    
    advisor = MarketingAdvisor()
    
    # 生成细分策略
    print("\nChampions 客户策略:")
    strategy = advisor.generate_segment_strategy('Champions')
    print(f"描述：{strategy['description']}")
    print(f"策略：{strategy['strategies'][:3]}")
    
    # 生成营销活动计划
    print("\n营销活动计划:")
    campaign = advisor.generate_campaign_plan(
        segments=['Champions', 'Loyal Customers', 'At Risk'],
        budget=100000,
        duration_days=30
    )
    print(f"活动名称：{campaign['name']}")
    print(f"目标细分：{len(campaign['target_segments'])} 个群体")
    print(f"预期 ROI: {campaign['expected_outcomes']['roi']}")
    
    # 生成 A/B 测试计划
    print("\nA/B 测试计划:")
    ab_test = advisor.generate_ab_test_plan(test_type='discount')
    print(f"测试名称：{ab_test['test_name']}")
    print(f"变体：{ab_test['variants']}")
    
    # 生成营销报告
    print("\n生成示例营销报告...")
    sample_data = pd.DataFrame({
        'segment': np.random.choice(['Champions', 'Loyal Customers', 'At Risk', 'Lost'], 100),
        'monetary': np.random.lognormal(3, 1, 100)
    })
    report = advisor.generate_marketing_report(rfm_data=sample_data)
    print(f"执行摘要：{report['executive_summary']}")
