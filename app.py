import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# 数据加载
@st.cache_data
def load_data():
    df = pd.read_csv("data/hongkong_real_estate.csv")  # 注意路径改为正斜杠
    return df

df = load_data()

# 页面配置
st.set_page_config(layout="wide")
st.title("🏠 香港房地产智能分析系统")

# ===================== 页面导航 =====================
page = st.sidebar.radio("导航", 
    ["📊 数据洞察", "📍 智能选址", "💰 房价预测"],
    horizontal=True
)

# ===================== 数据洞察页面 =====================
if page == "📊 数据洞察":
    st.header("📊 数据全景分析")
    
    # 1. 数据概览
    with st.expander("🔍 数据快照", expanded=True):
        st.dataframe(df.head(10))
    
    # 2. 动态分布可视化
    col1, col2 = st.columns(2)
    with col1:
        num_var = st.selectbox("选择数值变量", 
                             df.select_dtypes(include=np.number).columns)
        fig = px.histogram(df, x=num_var, nbins=30, 
                          title=f"{num_var}分布",
                          color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cat_var = st.selectbox("选择类别变量", 
                             df.select_dtypes(include=['object']).columns)
        fig = px.pie(df, names=cat_var, 
                    title=f"{cat_var}占比",
                    hole=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. 交互式散点矩阵
    st.subheader("🔗 变量关系探索")
    selected_vars = st.multiselect("选择分析变量", 
                                 df.select_dtypes(include=np.number).columns.tolist(),
                                 default=['price', 'size', 'flow'])
    if len(selected_vars) >= 2:
        fig = px.scatter_matrix(df, dimensions=selected_vars,
                               color='district', hover_name='usage')
        st.plotly_chart(fig, use_container_width=True)

# ===================== 智能选址页面 =====================
elif page == "📍 智能选址":
    st.header("📍 最优商铺选址推荐")
    
    with st.form("选址条件"):
        # 1. 筛选条件输入
        col1, col2 = st.columns(2)
        with col1:
            usage = st.selectbox("业态类型", df['usage'].unique())
            floor = st.selectbox("楼层偏好", df['floor'].unique())
            ac_type = st.selectbox("空调类型", df['air_conditioner_type'].unique())
        
        with col2:
            size_range = st.slider("面积需求(㎡)", *[int(df['size'].quantile(q)) for q in [0.1, 0.9]], (50, 200))
            price_range = st.slider("预算范围(元/㎡)", *[int(df['price'].quantile(q)) for q in [0.1, 0.9]], (10000, 50000))
        
        submitted = st.form_submit_button("开始智能推荐")
    
    if submitted:
        # 2. 动态筛选逻辑
        filtered = df[
            (df['usage'] == usage) &
            (df['floor'] == floor) &
            (df['air_conditioner_type'] == ac_type) &
            (df['size'].between(*size_range)) &
            (df['price'].between(*price_range))
        ]
        
        # 3. 智能推荐结果
        if not filtered.empty:
            # 计算各区域性价比指标
            district_stats = filtered.groupby('district').agg({
                'price': 'mean',
                'transportation_time': 'mean',
                'flow': 'mean',
                'size': 'count'
            }).sort_values('price')
            
            # 可视化展示
            tab1, tab2 = st.tabs(["推荐榜单", "空间分布"])
            
            with tab1:
                st.dataframe(
                    district_stats.style.format({
                        'price': '¥{:,.0f}',
                        'transportation_time': '{:.1f}分钟',
                        'flow': '{:,.0f}人'
                    }),
                    height=400
                )
            
            with tab2:
                fig = px.scatter(
                    filtered,
                    x='transportation_time',
                    y='price',
                    color='district',
                    size='size',
                    hover_data=['air_index', 'year'],
                    title="各区域价格-交通便利性分布"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("没有找到符合条件的商铺，建议放宽筛选条件")

# ===================== 房价预测页面 =====================
elif page == "💰 房价预测":
    st.header("💰 房价智能预测")
    
    with st.form("预测参数"):
        # 1. 参数输入
        col1, col2 = st.columns(2)
        with col1:
            district = st.selectbox("目标区域", df['district'].unique())
            usage = st.selectbox("业态类型", df['usage'].unique())
            size = st.number_input("面积(㎡)", min_value=10, value=100)
        
        with col2:
            trans_time = st.slider("地铁步行时间", 0, 60, 10)
            flow = st.slider("日均人流量", 0, 10000, 3000)
            age = st.slider("建筑年龄", 0, 50, 10)
        
        submitted = st.form_submit_button("获取预测")
    
    if submitted:
        # 2. 预测结果展示
        predicted_price = 8000 + size*1.8 - trans_time*8 + flow*0.08 - age*90
        
        # 可视化呈现
        st.subheader("📌 预测结果")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("预估单价", f"¥{predicted_price:,.0f}/㎡")
            st.metric("预估总价", f"¥{predicted_price*size:,.0f}")
            
            # 影响因素分析
            st.write("**主要影响因素:**")
            factors = pd.DataFrame({
                '因素': ['面积', '交通', '人流量', '房龄'],
                '影响系数': [1.8, -8, 0.08, -90]
            })
            st.dataframe(factors)
        
        with col2:
            # 类似房源比较
            similar = df[
                (df['district'] == district) & 
                (df['size'].between(size*0.8, size*1.2))
            ]
            
            if not similar.empty:
                fig = px.box(similar, y='price', 
                            title=f"{district}区域类似房源价格分布")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("该区域暂无类似房源参考")

# ===================== 页脚 =====================
st.markdown("---")
st.caption("© 2023 香港房地产分析平台 | 数据仅供参考")
