import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns
import os

# 假设已经加载了香港房价数据
@st.cache_data
def load_data():
    df = pd.read_csv("data\hongkong_real_estate.csv")  # 替换为香港房产数据路径
    return df

df = load_data()

st.title("🏠 香港房地产分析与推荐系统")

# 选择页面
page = st.sidebar.radio("选择功能页面", ["📊 描述性统计", "📍 推荐选址", "💰 预测房价"])

# ===================== 描述性统计分析 =====================
if page == "📊 描述性统计":
    st.header("📊 数据描述性统计")
    st.write("以下是基础统计描述：")
    st.dataframe(df.describe())

    st.subheader("📈 变量分布可视化")
    col = st.selectbox("选择数值变量", df.select_dtypes(include=[np.number]).columns)
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

# ===================== 推荐选址 =====================
elif page == "📍 推荐选址":
    st.header("📍 商铺推荐选址模型")

    # 输入框
    usage = st.selectbox("商铺类型 (usage)", df["usage"].unique())
    floor = st.selectbox("楼层分类 (floor)", df["floor"].unique())
    ac_type = st.selectbox("空调类型 (air_conditioner_type)", df["air_conditioner_type"].unique())
    air_quality = st.selectbox("空气质量指数", sorted(df["air_index"].unique()))

    # 范围选择
    size = st.slider("房屋面积（㎡）", 10, 1000, (50, 300))
    trans_time = st.slider("距离地铁站时间（分钟）", 0, 60, (0, 15))
    age = st.slider("房龄（年）", 0, 50, (0, 20))
    flow = st.slider("人流量", 0, 10000, (1000, 5000))
    price = st.slider("房屋单价（元/㎡）", 0, 200000, (5000, 30000))

    # 推荐区域
    st.subheader("推荐区域：")
    recommended = df[
        (df["usage"] == usage) &
        (df["floor"] == floor) &
        (df["air_conditioner_type"] == ac_type) &
        (df["air_index"] == air_quality) &
        (df["size"].between(*size)) &
        (df["transportation_time"].between(*trans_time)) &
        (df["year"].between(*age)) &
        (df["flow"].between(*flow)) &
        (df["price"].between(*price))
    ]

    if not recommended.empty:
        st.success("推荐区域如下：")
        st.dataframe(recommended["district"].value_counts().reset_index().rename(columns={"index": "district", "district": "count"}))
    else:
        st.warning("暂无满足条件的推荐结果")

# ===================== 房价预测 =====================
elif page == "💰 预测房价":
    st.header("💰 房价预测")

    usage = st.selectbox("商铺类型 (usage)", df["usage"].unique())
    floor = st.selectbox("楼层分类 (floor)", df["floor"].unique())
    ac_type = st.selectbox("空调类型 (air_conditioner_type)", df["air_conditioner_type"].unique())
    air_quality = st.selectbox("空气质量指数", sorted(df["air_index"].unique()))
    district = st.selectbox("区域 (district)", df["district"].unique())

    size = st.slider("房屋面积（㎡）", 10, 1000, 100)
    trans_time = st.slider("距离地铁站时间（分钟）", 0, 60, 10)
    age = st.slider("房龄（年）", 0, 50, 10)
    flow = st.slider("人流量", 0, 10000, 3000)

    # 简单预测模型
    def predict_price(size, trans_time, age, flow):
        return 5000 + size * 2 - trans_time * 10 + flow * 0.1 - age * 100

    predicted_price = predict_price(size, trans_time, age, flow)
    st.subheader("预测房价（元/㎡）:")
    st.success(f"🏷️ 预测结果为：{predicted_price:.2f} 元/㎡")

# ===================== 最优模型可视化 =====================
st.markdown("---")
st.markdown("📈 最优模型可视化展示")
st.image("your_model_plot.png", caption="示意图 - 最优模型结构图")  # 替换为你实际的模型图路径



# ========== 可选：退出按钮 ==========
if st.button("❌ 关闭程序（需终端操作）"):
    st.write("正在退出...")
    os._exit(0)  # 强制终止
