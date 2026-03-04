"""
客户细分分析平台 - 主界面
Streamlit 交互式 Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# 导入自定义模块
from rfm_analyzer import RFMAnalyzer, calculate_rfm_from_transactions
from clustering import CustomerClustering
from ltv_predictor import LTVPredictor
from churn_predictor import ChurnPredictor
from marketing_advisor import MarketingAdvisor

# 页面配置
st.set_page_config(
    page_title="客户细分分析平台",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def load_sample_data():
    """加载示例数据"""
    np.random.seed(42)
    n_customers = 1000
    
    # 生成客户交易数据
    customer_ids = range(1, n_customers + 1)
    
    # RFM 数据
    recency = np.random.exponential(60, n_customers).astype(int)
    frequency = np.random.poisson(5, n_customers) + 1
    monetary = np.random.lognormal(3.5, 1, n_customers)
    
    # 其他特征
    age = np.random.normal(35, 12, n_customers).astype(int)
    gender = np.random.choice(['M', 'F'], n_customers)
    region = np.random.choice(['East', 'West', 'South', 'North'], n_customers)
    
    # 创建流失标签 (基于 RFM)
    churn_prob = 1 / (1 + np.exp(-(recency/30 - frequency + np.log(monetary)/10)))
    churn = (np.random.random(n_customers) < churn_prob).astype(int)
    
    # 计算 LTV (简化公式)
    ltv = monetary * frequency * (12 / (recency/30 + 1)) * np.random.uniform(0.8, 1.2, n_customers)
    
    data = pd.DataFrame({
        'customer_id': customer_ids,
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'age': age,
        'gender': gender,
        'region': region,
        'churn': churn,
        'ltv': ltv
    })
    
    return data


def upload_data():
    """上传数据文件"""
    uploaded_file = st.file_uploader(
        "上传数据文件 (CSV/Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="支持 CSV 和 Excel 格式"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"成功加载 {len(df)} 条记录!")
            return df
        except Exception as e:
            st.error(f"加载文件失败：{str(e)}")
            return None
    
    return None


def main():
    """主函数"""
    
    # 标题
    st.markdown('<h1 class="main-header">📊 客户细分分析平台</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏
    st.sidebar.header("🎯 功能导航")
    
    # 初始化 session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'rfm_scores' not in st.session_state:
        st.session_state.rfm_scores = None
    if 'clusters' not in st.session_state:
        st.session_state.clusters = None
    if 'ltv_predictions' not in st.session_state:
        st.session_state.ltv_predictions = None
    if 'churn_predictions' not in st.session_state:
        st.session_state.churn_predictions = None
    
    # 数据加载部分
    st.sidebar.subheader("1️⃣ 数据加载")
    
    data_source = st.sidebar.radio(
        "选择数据源",
        ["示例数据", "上传文件"],
        help="选择使用内置示例数据或上传自己的数据"
    )
    
    if data_source == "示例数据":
        if st.session_state.data is None:
            st.session_state.data = load_sample_data()
            st.sidebar.success("示例数据已加载!")
    else:
        uploaded_df = upload_data()
        if uploaded_df is not None:
            st.session_state.data = uploaded_df
    
    # 显示数据预览
    if st.session_state.data is not None:
        with st.sidebar.expander("📋 数据预览", expanded=False):
            st.write(f"**数据维度**: {st.session_state.data.shape}")
            st.dataframe(st.session_state.data.head())
            
            # 列选择
            st.subheader("列映射")
            available_cols = st.session_state.data.columns.tolist()
            
            col_mapping = {
                'customer_id': st.selectbox("客户 ID", available_cols, index=0 if 'customer_id' in available_cols else 0),
                'recency': st.selectbox("最近购买 (天)", available_cols, index=1 if 'recency' in available_cols else 0),
                'frequency': st.selectbox("购买频率", available_cols, index=2 if 'frequency' in available_cols else 0),
                'monetary': st.selectbox("购买金额", available_cols, index=3 if 'monetary' in available_cols else 0),
            }
            
            st.session_state.col_mapping = col_mapping
    
    # 主功能选择
    st.sidebar.subheader("2️⃣ 分析功能")
    
    analysis_type = st.sidebar.selectbox(
        "选择分析类型",
        ["概览仪表板", "RFM 分析", "聚类分析", "LTV 预测", "流失预警", "营销建议"],
        help="选择要进行的分析类型"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 导出选项")
    
    export_format = st.sidebar.selectbox(
        "导出格式",
        ["CSV", "Excel", "PDF 报告"]
    )
    
    if st.sidebar.button("导出数据", type="primary"):
        if st.session_state.data is not None:
            st.sidebar.success("导出功能开发中...")
        else:
            st.sidebar.warning("请先加载数据")
    
    # 主内容区域
    if st.session_state.data is None:
        st.info("👈 请从左侧加载数据开始分析")
        return
    
    # 根据选择显示不同内容
    if analysis_type == "概览仪表板":
        show_dashboard_overview()
    elif analysis_type == "RFM 分析":
        show_rfm_analysis()
    elif analysis_type == "聚类分析":
        show_clustering_analysis()
    elif analysis_type == "LTV 预测":
        show_ltv_prediction()
    elif analysis_type == "流失预警":
        show_churn_prediction()
    elif analysis_type == "营销建议":
        show_marketing_advisor()


def show_dashboard_overview():
    """显示概览仪表板"""
    st.header("📈 概览仪表板")
    
    data = st.session_state.data
    
    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="总客户数",
            value=f"{len(data):,}",
            delta=None
        )
    
    with col2:
        if 'ltv' in data.columns:
            avg_ltv = data['ltv'].mean()
            st.metric(
                label="平均 LTV",
                value=f"¥{avg_ltv:,.0f}",
                delta=None
            )
        else:
            st.metric(
                label="平均 LTV",
                value="N/A",
                delta=None
            )
    
    with col3:
        if 'churn' in data.columns:
            churn_rate = data['churn'].mean() * 100
            st.metric(
                label="流失率",
                value=f"{churn_rate:.1f}%",
                delta=f"{-churn_rate:.1f}%",
                delta_color="inverse"
            )
        else:
            st.metric(
                label="流失率",
                value="N/A",
                delta=None
            )
    
    with col4:
        if 'monetary' in data.columns:
            avg_monetary = data['monetary'].mean()
            st.metric(
                label="平均客单价",
                value=f"¥{avg_monetary:,.0f}",
                delta=None
            )
        else:
            st.metric(
                label="平均客单价",
                value="N/A",
                delta=None
            )
    
    st.markdown("---")
    
    # 图表行 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("客户分布 - 地区")
        if 'region' in data.columns:
            region_dist = data['region'].value_counts()
            fig = px.pie(
                values=region_dist.values,
                names=region_dist.index,
                title="地区分布"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("数据中无地区信息")
    
    with col2:
        st.subheader("客户分布 - 性别")
        if 'gender' in data.columns:
            gender_dist = data['gender'].value_counts()
            fig = px.bar(
                x=gender_dist.index,
                y=gender_dist.values,
                labels={'x': '性别', 'y': '客户数'},
                title="性别分布"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("数据中无性别信息")
    
    # 图表行 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RFM 指标分布")
        rfm_cols = ['recency', 'frequency', 'monetary']
        available_rfm = [col for col in rfm_cols if col in data.columns]
        
        if available_rfm:
            fig = px.histogram(
                data[available_rfm],
                nbins=30,
                title="RFM 指标分布",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("数据中无 RFM 信息")
    
    with col2:
        st.subheader("LTV 分布")
        if 'ltv' in data.columns:
            fig = px.histogram(
                data,
                x='ltv',
                nbins=50,
                title="LTV 分布直方图",
                labels={'ltv': '生命周期价值 (¥)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("数据中无 LTV 信息")


def show_rfm_analysis():
    """显示 RFM 分析"""
    st.header("🎯 RFM 客户价值分析")
    
    data = st.session_state.data.copy()
    
    # 检查必要列
    required_cols = ['recency', 'frequency', 'monetary']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        st.error(f"数据中缺少必要列：{missing_cols}")
        return
    
    # RFM 分析
    st.subheader("1. RFM 分数计算")
    
    analyzer = RFMAnalyzer(data)
    rfm_scores = analyzer.calculate_rfm_scores()
    segments = analyzer.segment_customers()
    
    st.session_state.rfm_scores = segments
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("平均 R 分数", f"{rfm_scores['R_score'].mean():.2f}")
    
    with col2:
        st.metric("平均 F 分数", f"{rfm_scores['F_score'].mean():.2f}")
    
    with col3:
        st.metric("平均 M 分数", f"{rfm_scores['M_score'].mean():.2f}")
    
    st.markdown("---")
    
    # 细分群体分布
    st.subheader("2. 客户细分分布")
    
    segment_dist = analyzer.get_segment_distribution()
    
    fig = px.bar(
        x=list(segment_dist.keys()),
        y=list(segment_dist.values()),
        labels={'x': '细分群体', 'y': '客户数'},
        title="各细分群体客户数量",
        color=list(segment_dist.keys()),
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 细分群体统计
    st.subheader("3. 细分群体特征")
    
    summary = analyzer.get_segment_summary()
    st.dataframe(summary)
    
    # 细分群体详情
    st.subheader("4. 各群体详细分析")
    
    selected_segment = st.selectbox(
        "选择细分群体",
        options=sorted(segment_dist.keys())
    )
    
    segment_data = segments[segments['segment'] == selected_segment]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("群体客户数", f"{len(segment_data):,}")
    
    with col2:
        st.metric("平均 RFM 总分", f"{segment_data['RFM_score'].mean():.1f}")
    
    with col3:
        st.metric("占比", f"{len(segment_data)/len(segments)*100:.1f}%")
    
    # 雷达图
    st.subheader("5. 群体特征雷达图")
    
    # 准备雷达图数据
    segment_profiles = segments.groupby('segment')[['recency', 'frequency', 'monetary']].mean()
    
    # 标准化
    segment_profiles_norm = (segment_profiles - segment_profiles.min()) / (segment_profiles.max() - segment_profiles.min() + 1e-10)
    
    fig = go.Figure()
    
    for segment in segment_profiles_norm.index:
        values = segment_profiles_norm.loc[segment].tolist()
        values += values[:1]  # 闭合图形
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=['Recency', 'Frequency', 'Monetary', 'Recency'],
            fill='toself',
            name=segment
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title='各细分群体 RFM 特征对比',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_clustering_analysis():
    """显示聚类分析"""
    st.header("🔍 客户聚类分析")
    
    data = st.session_state.data.copy()
    
    # 特征选择
    st.subheader("1. 特征选择")
    
    available_features = ['recency', 'frequency', 'monetary', 'age', 'ltv']
    selected_features = [col for col in available_features if col in data.columns]
    
    if not selected_features:
        st.error("数据中没有可用的数值特征列")
        return
    
    feature_selection = st.multiselect(
        "选择用于聚类的特征",
        options=selected_features,
        default=selected_features[:3] if len(selected_features) >= 3 else selected_features
    )
    
    if len(feature_selection) < 2:
        st.warning("请至少选择 2 个特征进行聚类")
        return
    
    # 寻找最优 K
    st.subheader("2. 最优 K 值选择")
    
    clusterer = CustomerClustering(data, features=feature_selection)
    
    k_range = st.slider("K 值搜索范围", 2, 10, (2, 8))
    metrics = clusterer.find_optimal_k(range(k_range[0], k_range[1] + 1))
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_elbow = clusterer.plot_elbow_curve(metrics)
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col2:
        fig_silhouette = clusterer.plot_silhouette_analysis(metrics)
        st.plotly_chart(fig_silhouette, use_container_width=True)
    
    # 执行聚类
    st.subheader("3. 执行聚类")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.number_input("聚类数量", min_value=2, max_value=10, value=4)
    
    with col2:
        clustering_method = st.selectbox(
            "聚类方法",
            ["K-Means", "层次聚类", "DBSCAN"]
        )
    
    with col3:
        if st.button("执行聚类", type="primary"):
            if clustering_method == "K-Means":
                labels = clusterer.kmeans_clustering(n_clusters=n_clusters)
            elif clustering_method == "层次聚类":
                labels = clusterer.hierarchical_clustering(n_clusters=n_clusters)
            else:  # DBSCAN
                labels = clusterer.dbscan_clustering(eps=0.5, min_samples=5)
            
            st.session_state.clusters = labels
            st.session_state.clusterer = clusterer
            st.success(f"聚类完成！共 {len(set(labels))} 个簇")
    
    # 显示聚类结果
    if st.session_state.clusters is not None:
        st.subheader("4. 聚类结果可视化")
        
        # 更新数据
        data['cluster'] = st.session_state.clusters
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = clusterer.plot_cluster_distribution(feature1=feature_selection[0], 
                                                           feature2=feature_selection[1] if len(feature_selection) > 1 else None)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            fig_radar = clusterer.plot_radar_chart()
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # 聚类统计
        st.subheader("5. 聚类统计")
        
        stats = clusterer.get_cluster_statistics()
        st.dataframe(stats)
        
        # 聚类画像
        st.subheader("6. 聚类画像")
        
        profiles = clusterer.get_cluster_profiles()
        st.dataframe(profiles)


def show_ltv_prediction():
    """显示 LTV 预测"""
    st.header("💰 客户生命周期价值 (LTV) 预测")
    
    data = st.session_state.data.copy()
    
    # 检查 LTV 列
    if 'ltv' not in data.columns:
        st.warning("数据中没有 LTV 列，将使用 RFM 指标估算 LTV")
        # 简单估算 LTV
        data['ltv'] = data['monetary'] * data['frequency'] * (12 / (data['recency'] / 30 + 1))
    
    # 特征选择
    st.subheader("1. 特征选择")
    
    available_features = ['recency', 'frequency', 'monetary', 'age']
    selected_features = [col for col in available_features if col in data.columns]
    
    predictor = LTVPredictor(data, target_col='ltv')
    predictor.prepare_features(feature_cols=selected_features)
    
    # 模型训练
    st.subheader("2. 模型训练")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "选择模型",
            ["线性回归", "Ridge 回归", "随机森林", "梯度提升"]
        )
    
    with col2:
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)
    
    with col3:
        if st.button("训练模型", type="primary"):
            if model_type == "线性回归":
                predictor.train_linear_model(regularization='linear')
            elif model_type == "Ridge 回归":
                predictor.train_linear_model(regularization='ridge')
            elif model_type == "随机森林":
                predictor.train_random_forest()
            else:
                predictor.train_gradient_boosting()
            
            st.session_state.ltv_predictor = predictor
            st.success("模型训练完成!")
    
    # 模型评估
    if hasattr(st.session_state, 'ltv_predictor'):
        st.subheader("3. 模型评估")
        
        comparison = predictor.get_model_comparison()
        st.dataframe(comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_importance = predictor.plot_feature_importance()
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            fig_pred_vs_actual = predictor.plot_predictions_vs_actual()
            st.plotly_chart(fig_pred_vs_actual, use_container_width=True)
        
        # LTV 分布
        st.subheader("4. LTV 预测分布")
        
        predictions = predictor.predict()
        ltv_dist = predictor.calculate_ltv_distribution(predictions)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("平均 LTV", f"¥{ltv_dist['mean']:,.0f}")
        
        with col2:
            st.metric("中位数 LTV", f"¥{ltv_dist['median']:,.0f}")
        
        with col3:
            st.metric("LTV 标准差", f"¥{ltv_dist['std']:,.0f}")
        
        with col4:
            st.metric("P90 LTV", f"¥{ltv_dist['q90']:,.0f}")
        
        # LTV 直方图
        fig = px.histogram(
            x=predictions,
            nbins=50,
            title="LTV 预测分布",
            labels={'x': '预测 LTV (¥)'}
        )
        st.plotly_chart(fig, use_container_width=True)


def show_churn_prediction():
    """显示流失预测"""
    st.header("⚠️ 客户流失预警")
    
    data = st.session_state.data.copy()
    
    # 检查流失标签
    if 'churn' not in data.columns:
        st.warning("数据中没有流失标签列")
        st.info("流失标签应为 0/1 二值变量，0=未流失，1=流失")
        return
    
    # 特征选择
    st.subheader("1. 特征选择")
    
    available_features = ['recency', 'frequency', 'monetary', 'age', 'ltv']
    selected_features = [col for col in available_features if col in data.columns]
    
    predictor = ChurnPredictor(data, target_col='churn')
    predictor.prepare_features(feature_cols=selected_features)
    
    # 流失率统计
    churn_rate = data['churn'].mean() * 100
    st.metric("当前流失率", f"{churn_rate:.1f}%")
    
    # 模型训练
    st.subheader("2. 模型训练")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_type = st.selectbox(
            "选择模型",
            ["逻辑回归", "随机森林", "梯度提升"]
        )
    
    with col2:
        threshold = st.slider("流失风险阈值", 0.3, 0.9, 0.5, 0.05)
    
    with col3:
        if st.button("训练模型", type="primary"):
            if model_type == "逻辑回归":
                predictor.train_logistic_regression()
            elif model_type == "随机森林":
                predictor.train_random_forest()
            else:
                predictor.train_gradient_boosting()
            
            st.session_state.churn_predictor = predictor
            st.success("模型训练完成!")
    
    # 模型评估
    if hasattr(st.session_state, 'churn_predictor'):
        st.subheader("3. 模型评估")
        
        comparison = predictor.get_model_comparison()
        st.dataframe(comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_roc = predictor.plot_roc_curve()
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            fig_cm = predictor.plot_confusion_matrix()
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # 特征重要性
        st.subheader("4. 流失影响因素")
        
        fig_importance = predictor.plot_feature_importance(top_n=10)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # 高风险客户
        st.subheader("5. 高流失风险客户")
        
        high_risk = predictor.get_high_risk_customers(threshold=threshold)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("高风险客户数", f"{len(high_risk):,}")
        
        with col2:
            st.metric("高风险占比", f"{len(high_risk)/len(data)*100:.1f}%")
        
        st.dataframe(high_risk.head(20))


def show_marketing_advisor():
    """显示营销建议"""
    st.header("📢 智能营销建议")
    
    advisor = MarketingAdvisor()
    
    # 检查是否有 RFM 数据
    if st.session_state.rfm_scores is None:
        st.warning("请先进行 RFM 分析以获取客户细分")
        st.info("切换到 'RFM 分析' 标签页进行分析")
        
        # 使用示例策略
        st.subheader("通用营销策略建议")
        
        segments = ['Champions', 'Loyal Customers', 'New Customers', 'At Risk']
        
        for segment in segments:
            with st.expander(f"🎯 {segment} 策略"):
                strategy = advisor.generate_segment_strategy(segment)
                st.write(f"**描述**: {strategy['description']}")
                st.write("**推荐策略**:")
                for s in strategy['strategies']:
                    st.write(f"- {s}")
    else:
        # 基于 RFM 数据的营销建议
        st.subheader("1. 客户细分概览")
        
        segment_dist = st.session_state.rfm_scores['segment'].value_counts().to_dict()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总客户数", f"{len(st.session_state.rfm_scores):,}")
        
        with col2:
            high_value = segment_dist.get('Champions', 0) + segment_dist.get('Loyal Customers', 0)
            st.metric("高价值客户", f"{high_value:,}")
        
        with col3:
            at_risk = segment_dist.get('At Risk', 0) + segment_dist.get('Lost', 0)
            st.metric("风险客户", f"{at_risk:,}")
        
        # 细分群体策略
        st.subheader("2. 各群体营销策略")
        
        selected_segment = st.selectbox(
            "选择细分群体查看策略",
            options=sorted(segment_dist.keys())
        )
        
        strategy = advisor.generate_segment_strategy(
            selected_segment,
            st.session_state.rfm_scores[st.session_state.rfm_scores['segment'] == selected_segment]
        )
        
        st.write(f"**群体描述**: {strategy['description']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**特征**:")
            for char in strategy.get('characteristics', []):
                st.write(f"- {char}")
        
        with col2:
            st.write("**推荐渠道**:")
            for channel in strategy.get('channels', []):
                st.write(f"- {channel}")
        
        st.write("**推荐策略**:")
        for s in strategy.get('strategies', []):
            st.write(f"✅ {s}")
        
        # 营销活动计划
        st.subheader("3. 营销活动计划生成")
        
        target_segments = st.multiselect(
            "选择目标细分群体",
            options=sorted(segment_dist.keys()),
            default=['Champions', 'Loyal Customers']
        )
        
        budget = st.number_input("活动预算 (元)", min_value=0, value=100000, step=10000)
        duration = st.slider("活动持续天数", 7, 90, 30)
        
        if st.button("生成营销计划", type="primary"):
            if target_segments:
                campaign = advisor.generate_campaign_plan(
                    segments=target_segments,
                    budget=budget,
                    duration_days=duration
                )
                
                st.success("营销计划生成完成!")
                
                st.write(f"**活动名称**: {campaign['name']}")
                st.write(f"**持续时间**: {campaign['duration_days']} 天")
                st.write(f"**总预算**: ¥{campaign['total_budget']:,.0f}")
                
                # 时间线
                st.write("**活动时间线**:")
                for phase in campaign['timeline']:
                    with st.expander(f"📅 {phase['phase']} (第{phase['start_day']}-{phase['end_day']}天)"):
                        for activity in phase['activities']:
                            st.write(f"- {activity}")
                
                # 预期结果
                st.write("**预期效果**:")
                for metric, value in campaign['expected_outcomes'].items():
                    st.write(f"- {metric}: {value}")
        
        # A/B 测试建议
        st.subheader("4. A/B 测试建议")
        
        test_type = st.selectbox(
            "选择测试类型",
            ["折扣力度测试", "邮件主题行测试", "行动号召测试", "发送时间测试"]
        )
        
        if st.button("生成 A/B 测试方案"):
            test_map = {
                "折扣力度测试": "discount",
                "邮件主题行测试": "subject_line",
                "行动号召测试": "cta",
                "发送时间测试": "timing"
            }
            
            ab_test = advisor.generate_ab_test_plan(test_type=test_map[test_type])
            
            st.write(f"**测试名称**: {ab_test['test_name']}")
            st.write(f"**测试变体**: {' vs '.join(ab_test['variants'])}")
            st.write(f"**样本量**: {ab_test['sample_size']}")
            st.write(f"**持续时间**: {ab_test['duration']}")
            st.write(f"**评估指标**: {', '.join(ab_test['metrics'])}")


if __name__ == "__main__":
    main()
