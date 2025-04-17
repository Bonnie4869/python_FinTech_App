import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from math import floor
from joblib import load
import os
import plotly.express as px

#page 1
def show_statistics_analysis():
    st.title("📈 Hong Kong Store Rental Statistics Analysis")

    # Load data
    @st.cache_data
    def load_data():
        try:
            data = pd.read_excel("data/stores_rent.xlsx")
            # Add any necessary data preprocessing here
            return data
        except:
            return pd.DataFrame()
    data = load_data()
    
    if data.empty:
        st.warning("No data available. Please check your data file.")
        return
    
    #sunburst graph
    def plot_district_sunburst(data):
        district_counts = data.groupby(['district1', 'district2', 'district3']).size().reset_index(name='count')
        
        fig = px.sunburst(
            district_counts,
            path=['district1', 'district2', 'district3'],  
            values='count',                                
            color='count',                                
            color_continuous_scale='Blues',                
            title='Distribution Map (Region from big to small)',
            width=800,
            height=600
        )
        
        fig.update_traces(
            textinfo="label+percent parent",  
            insidetextorientation='radial'    
        )
        fig.update_layout(
            margin=dict(t=40, b=0, l=0, r=0),
            coloraxis_colorbar=dict(title='Number of Store')
        )
        
        return fig

    # Section 1: Key Metrics
    st.header("1. Key Market Metrics")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Avg Monthly Rent", f"HK${data['avg_price_monthly'].mean():,.0f}")
    with cols[1]:
        st.metric("Median Size", f"{floor(data['area(foot)'].median())}")  # 使用floor取整
    with cols[2]:
        st.metric("Top District", data['district1'].mode()[0])
    
    # Section 2: Rent Distribution
    st.header("2. Rent Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['avg_price_monthly'], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Monthly Rent (HKD)")
    st.pyplot(fig)
    
    # Section 3: District Analysis
    st.header("3. District Analysis")

    district_prices = data.groupby('district1')['avg_price_monthly'].mean().sort_values(ascending=False)
        
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=district_prices.index,
        y=district_prices.values,
        hue=district_prices.index,  # Fix here
        palette="viridis",
        ax=ax,
        legend=False  # Fix here
    )
    ax.set_title("Average Monthly Rent by District")
    ax.set_xlabel("District")
    ax.set_ylabel("Rent (HKD/month)")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)
    
    # Section 4: Top Price Factors (FIXED)
    st.header("4. Top Price Factors")
    numeric_cols = data.select_dtypes(include=np.number).columns
    if 'avg_price_monthly' in numeric_cols:
        corr = data[numeric_cols].corr()['avg_price_monthly'].sort_values(key=abs, ascending=False)[1:4]
        st.write("Correlation with rent price:")
        
        for factor, score in corr.items():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.write(f"{factor}: {score:.2f}")
            with col2:
                color = "green" if score > 0 else "red"
                st.markdown(f"""
                <div style="background: #f0f0f0; border-radius: 5px; padding: 2px;">
                    <div style="background: {color}; width: {abs(score)*100}%; 
                                border-radius: 5px; text-align: center; color: white;">
                        {score:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Section 5: Usage Comparison
    st.header("5. Usage Type Comparison")

    if 'usage' in data.columns:
        usage_prices = data.groupby('usage')['avg_price_monthly'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x=usage_prices.index, y=usage_prices.values, ax=ax)
        ax.set_ylabel("Monthly Rent (HKD)")  
        ax.tick_params(axis='x', rotation=45)  
        st.pyplot(fig)
    else:
        st.warning("Usage data not available")

    # Section 6: District Hierarchy Sunburst
    st.header("6. Hierarchical distribution of regional housing resources")

    district_hierarchy = {
        "HK_Island": ["Cen_West", "WanChai", "East", "South"],
        "Kowloon": ["YauTsimMong", "ShamShuiPo", "Kowloon_City", "WongTaiSin", "KwunTong"],
        "NT_East": ["ShaTin", "TaiPo", "North"],
        "NT_West": ["TsuenWan", "TuenMun", "YuenLong", "KwaiTsing"]
    }

    district3_mapping = {
        # HK Island
        "Cen_West": ["Central", "Admiralty", "SheungWan"],
        "WanChai": ["WanChai", "Causeway_Bay", "Happy_Valley", "TinHau"],
        "East": ["North_Point", "Quarry_Bay", "ChaiWan", "ShauKeiWan", "TaikooShing"],
        "South": ["Aberdeen", "Southern"],
        
        # Kowloon
        "YauTsimMong": ["TsimShaTsui", "YauMaTei", "Jordan", "MongKok", "TaiKokTsui"],
        "ShamShuiPo": ["ShamShuiPo", "CheungShaWan", "LaiChiKok"],
        "Kowloon_City": ["HungHom", "Kowloon_Bay", "SanPoKong", "TokwaWan"],
        "WongTaiSin": ["WongTaiSin"],
        "KwunTong": ["KwunTong"],
        
        # NT_East
        "ShaTin": ["ShaTin", "TaiWai"],
        "TaiPo": ["TaiPo"],
        "North": ["Fanling", "SheungShui"],
        
        # NT_West
        "TsuenWan": ["TsuenWan", "KwaiChung"],
        "TuenMun": ["TuenMun"],
        "YuenLong": ["YuenLong"],
        "KwaiTsing": ["KwaiTsing"]
    }

    if all(col in data.columns for col in ['district1', 'district2', 'district3']):
        valid_data = data[
            data['district2'].isin([d2 for d1 in district_hierarchy.values() for d2 in d1]) &
            data['district3'].isin([d3 for d2 in district3_mapping.values() for d3 in d2])
        ]
        
        if len(valid_data) > 0:
            fig = plot_district_sunburst(valid_data)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("error")
            fig = plot_district_sunburst(data)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("error")

#page 2
def show_predict_page():
    st.title("HK Store Rent Prediction")
    
    @st.cache_resource
    def load_model():
        model_path = 'models/predicted_price.pkl'
        if not os.path.exists(model_path):
            st.error(f"Error: Model file not found at {os.path.abspath(model_path)}")
            st.stop()
        return load(model_path)

    model = load_model()


    st.header("1️⃣ Store basic information")

    col1, col2 = st.columns(2)
    with col1:
        usage = st.selectbox("Select the type of the store", ["Office", "Industrial"])
    with col2:
        floor = st.selectbox("Select the floor", ['G/F', 'Low', 'Mid', 'High'])
    floor_mapping = {'G/F': 1, 'Low': 2, 'Mid': 3, 'High': 4}


    col1, col2, col3 = st.columns(3)

    with col1:
        district1 = st.selectbox("select the region", ['HK_Island', 'Kowloon', 'NT_East', 'NT_West'])

    with col2:
        district2_options = {
            'HK_Island': ['Cen_West', 'WanChai', 'East', 'South'],
            'Kowloon': ['YauTsimMong', 'ShamShuiPo', 'Kowloon_City', 'WongTaiSin', 'KwunTong'],
            'NT_East': ['ShaTin', 'TaiPo', 'North'],
            'NT_West': ['TsuenWan', 'TuenMun', 'YuenLong', 'KwaiTsing']
        }
        district2 = st.selectbox("select the district", district2_options[district1])

    with col3:
        hierarchy = {
            # HK Island
            "Cen_West": ["Central", "Admiralty", "SheungWan"],
            "WanChai": ["WanChai", "Causeway_Bay", "Happy_Valley", "TinHau"],
            "East": ["North_Point", "Quarry_Bay", "ChaiWan", "ShauKeiWan", "TaikooShing"],
            "South": ["Aberdeen", "Southern"],
            
            # Kowloon
            "YauTsimMong": ["TsimShaTsui", "YauMaTei", "Jordan", "MongKok", "TaiKokTsui", "PrinceEdward"],
            "ShamShuiPo": ["ShamShuiPo", "CheungShaWan", "LaiChiKok"],
            "Kowloon_City": ["HungHom", "Kowloon_Bay", "SanPoKong", "TokwaWan"],
            "WongTaiSin": ["WongTaiSin"],
            "KwunTong": ["KwunTong"],
            
            # NT_East
            "ShaTin": ["ShaTin", "TaiWai"],
            "TaiPo": ["TaiPo"],
            "North": ["Fanling", "SheungShui"],
            
            # NT_West
            "TsuenWan": ["TsuenWan", "KwaiChung"],
            "TuenMun": ["TuenMun"],
            "YuenLong": ["YuenLong"],
            "KwaiTsing": ["KwaiTsing"]
        }

        if district2 in hierarchy:
            district3 = st.selectbox("Detailed places", hierarchy[district2])
        else:
            district3 = None



    col1, col2 = st.columns(2)
    with col1:
        ac_type = st.selectbox("Select the type of air conditioner", ['central air-conditioning', 'chilled water system', 'hybrid air-conditioning', 'individual air-conditioning', 'split-type air conditioner', 'Others'])
    with col2:
        area = st.number_input("Area (Square Feet)", min_value=50, max_value=100000, value=2000, step=50)


    st.header("2️⃣ Surrounding environment")
    col1, col2 = st.columns(2)
    with col1:
        transport_time = st.slider("Distance to MTR Station (minutes)", min_value=1, max_value=30, value=5, step=1)
        age = st.slider("Store age (Year)", min_value=1, max_value=120, value=60, step=1)
    with col2:
        normalized_value = st.slider(
            "People density (0-100%)", 
            min_value=0, 
            max_value=100, 
            value=60,
            step=1
        )
        vendor_points = int(normalized_value / 100 * (2400 - 1900) + 1900)
        aqi = st.slider("Air Quality Index (AQI)", min_value=20, max_value=80, value=50)


    district1_mapping = {'HK_Island': 1, 'Kowloon': 2, 'NT_East': 3, 'NT_West': 4}
    district2_mapping = {
        'Cen_West': 1, 'WanChai': 2, 'East': 3, 'South': 4,
        'YauTsimMong': 1, 'ShamShuiPo': 2, 'Kowloon_City': 3, 'WongTaiSin': 4, 'KwunTong': 5,
        'ShaTin': 1, 'TaiPo': 2, 'North': 3,
        'TsuenWan': 1, 'TuenMun': 2, 'YuenLong': 3, 'KwaiTsing': 4
    }

    district3_code_mapping = {}
    for district2_name, district3_list in hierarchy.items():
        if district2_name in ["Cen_West", "WanChai", "East", "South"]:
            district1 = "HK_Island"
        elif district2_name in ["YauTsimMong", "ShamShuiPo", "Kowloon_City", "WongTaiSin", "KwunTong"]:
            district1 = "Kowloon"
        elif district2_name in ["ShaTin", "TaiPo", "North"]:
            district1 = "NT_East"
        else:
            district1 = "NT_West"
        
        for i, district3 in enumerate(district3_list, 1):
            code = district1_mapping[district1] * 10000 + district2_mapping[district2_name] * 100 + i
            district3_code_mapping[district3] = code

    if st.button("Predict", type="primary"):
        if district3 is None:
            st.error("Please select a valid detailed location")
            return
            
        features = {
            'floor_encoded': floor_mapping[floor],
            'AQI': aqi,
            'vendor_points': vendor_points,
            'store_age': age,
            'transportation_time(min)': transport_time,
            'area(foot)': area,
            'Industrial': 1 if usage == "Industrial" else 0,
            'Office': 1 if usage == "Office" else 0,
            'district_code': district3_code_mapping.get(district3, 10101),
            'Cen_air_con': 1 if ac_type == 'central air-conditioning' else 0, 
            'Chilled_Water_sys': 1 if ac_type == 'chilled water system' else 0,
            'Hybrid_air_con': 1 if ac_type == 'hybrid air-conditioning' else 0,
            'Indi_air_con': 1 if ac_type == 'individual air-conditioning' else 0,
            'Split_air_con': 1 if ac_type == 'split-type air conditioner' else 0, 
            'Others': 1 if ac_type == 'Others' else 0,
        }
        
        input_df = pd.DataFrame([features])[['floor_encoded', 'AQI', 'vendor_points', 'store_age',
                                           'transportation_time(min)', 'area(foot)', 
                                           'Industrial', 'Office', 'district_code',
                                           'Cen_air_con', 'Chilled_Water_sys','Hybrid_air_con', 'Indi_air_con', 'Split_air_con', 'Others']]
        
        prediction = model.predict(input_df)
        st.success(f"Predicted price: HK${prediction[0]:,.2f}")
        
        # with st.expander("View input features (for testing)"):
        #     st.json(features)

#page 3
def show_recommended_location():
    st.title("🏪 HK Store Location Recommendation System")

    # 加载模型
    @st.cache_resource
    def load_model():
        model_path = 'models/decision_tree.pkl'
        if not os.path.exists(model_path):
            st.error(f"error: {os.path.abspath(model_path)}")
            st.stop()
        return load(model_path)

    model = load_model()


    st.header("1️⃣ Store basic information")


    col1, col2 = st.columns(2)
    with col1:
        usage = st.selectbox("Select the type of the usage", ["Office", "Commercial"])
    with col2:
        floor = st.selectbox("Select the floor", ['G/F', 'Low', 'Mid', 'High'])
    floor_mapping = {'G/F': 1, 'Low': 2, 'Mid': 3, 'High': 4}

    col1, col2, col3= st.columns(3)
    with col1:
        area = st.number_input("Area (Square Feet)", min_value=50, max_value=100000, value=800, step=50)
        age = st.slider("Store age (Years)", min_value=1, max_value=120, value=10, step=1)
    with col2:
        transport_time = st.slider("Distance to MTR Station (minutes)", min_value=1, max_value=30, value=10, step=1)
        avg_price = st.number_input("Average Monthly Rent (HKD)", min_value=5000, max_value=500000, value=30000, step=1000)
    with col3:
        ac_type = st.selectbox("Select the type of air conditioner", ['central air-conditioning', 'chilled water system', 'hybrid air-conditioning', 'individual air-conditioning', 'split-type air conditioner', 'Others'])


    st.header("2️⃣ Surrounding environment")
    col1, col2 = st.columns(2)
    with col1:
        normalized_value = st.slider(
            "People Density (0-100%)", 
            min_value=0, 
            max_value=100, 
            value=60,
            step=1
        )
        vendor_points = int(normalized_value / 100 * (2400 - 1900) + 1900)
    with col2:
        aqi = st.slider("Air Quality Index (AQI)", min_value=20, max_value=80, value=50)

    if st.button("🚀 Get Location Recommendations", type="primary", use_container_width=True):
        features = {
            'floor_encoded': floor_mapping[floor],
            'AQI': aqi,
            'vendor_points': vendor_points,
            'store_age': age,
            'transportation_time(min)': transport_time,
            'area(foot)': area,
            'Industrial': 1 if usage == "Commercial" else 0,
            'Office': 1 if usage == "Office" else 0,
            'avg_price_monthly': avg_price,
            'Cen_air_con': 1 if ac_type == 'central air-conditioning' else 0, 
            'Chilled_Water_sys': 1 if ac_type == 'chilled water system' else 0,
            'Hybrid_air_con': 1 if ac_type == 'hybrid air-conditioning' else 0,
            'Indi_air_con': 1 if ac_type == 'individual air-conditioning' else 0,
            'Split_air_con': 1 if ac_type == 'split-type air conditioner' else 0, 
            'Others': 1 if ac_type == 'Others' else 0,
        }
        
        # Create DataFrame ensuring correct column order
        input_df = pd.DataFrame([features], columns=[
            'floor_encoded', 'AQI', 'vendor_points', 'store_age',
            'transportation_time(min)', 'area(foot)', 'Industrial', 
            'Office', 'avg_price_monthly',
            'Cen_air_con', 'Chilled_Water_sys','Hybrid_air_con', 'Indi_air_con', 'Split_air_con', 'Others',
        ])
        
        try:
            with st.spinner("🔍 Analyzing best locations..."):
                # Pass the DataFrame to predict
                predicted_district_code = model.predict(input_df)[0]
            
            # District encoding mapping
            district1_mapping = {
                'HK_Island': 1,
                'Kowloon': 2,
                'NT_East': 3,
                'NT_West': 4
            }

            district2_mapping = {
                # Hong Kong Island HK_Island (district1=1)
                'Cen_West': 1,  # Central and Western District
                'WanChai': 2,   # Wan Chai District
                'East': 3,      # Eastern District
                'South': 4,     # Southern District

                # Kowloon (district1=2)
                'YauTsimMong': 1,  # Yau Tsim Mong District
                'ShamShuiPo': 2,   # Sham Shui Po District
                'Kowloon_City': 3, # Kowloon City District
                'WongTaiSin': 4,   # Wong Tai Sin District
                'KwunTong': 5,     # Kwun Tong District

                # New Territories East NT_East (district1=3)
                'ShaTin': 1,       # Sha Tin District
                'TaiPo': 2,        # Tai Po District
                'North': 3,        # North District

                # New Territories West NT_West (district1=4)
                'TsuenWan': 1,     # Tsuen Wan District
                'TuenMun': 2,      # Tuen Mun District
                'YuenLong': 3,     # Yuen Long District
                'KwaiTsing': 4     # Kwai Tsing District
            }

            hierarchy = {
                # Hong Kong Island
                "Cen_West": {
                    "district1": "HK_Island",
                    "district3": ["Central", "Admiralty", "SheungWan"]
                },
                "WanChai": {
                    "district1": "HK_Island",
                    "district3": ["WanChai", "Causeway_Bay", "Happy_Valley", "TinHau"]
                },
                "East": {
                    "district1": "HK_Island",
                    "district3": ["North_Point", "Quarry_Bay", "ChaiWan", "ShauKeiWan", "TaikooShing"]
                },
                "South": {
                    "district1": "HK_Island",
                    "district3": ["Aberdeen", "Southern"]
                },

                # Kowloon
                "YauTsimMong": {
                    "district1": "Kowloon",
                    "district3": ["TsimShaTsui", "YauMaTei", "Jordan", "MongKok", "TaiKokTsui"]
                },
                "ShamShuiPo": {
                    "district1": "Kowloon",
                    "district3": ["CheungShaWan", "Prince"]
                },
                "Kowloon_City": {
                    "district1": "Kowloon",
                    "district3": ["HungHom", "Kowloon_Bay", "SanPoKong", "TokwaWan"]
                },
                "KwunTong": {
                    "district1": "Kowloon",
                    "district3": ["KwunTong"]
                },
                "WongTaiSin": {
                    "district1": "Kowloon",
                    "district3": ["WongTaiSin"]
                },

                # New Territories East
                "ShaTin": {
                    "district1": "NT_East",
                    "district3": ["ShaTin", "TaiWai"]
                },
                "TaiPo": {
                    "district1": "NT_East",
                    "district3": ["TaiPo", "Fanling", "SheungShui"]
                },
                "North":{
                    "district1": "NT_East",
                    "district3": ["North"]
                },

                # New Territories West
                "TsuenWan": {
                    "district1": "NT_West",
                    "district3": ["TsuenWan", "KwaiChung"]
                },
                "TuenMun": {
                    "district1": "NT_West",
                    "district3": ["TuenMun"]
                },
                "YuenLong": {
                    "district1": "NT_West",
                    "district3": ["YuenLong"]
                }
            }


            def decode_district(code):
                try:
                    code_str = str(code).zfill(5)
                    district1_code = int(code_str[0])
                    district2_code = int(code_str[1:3])
                    district3_code = int(code_str[3:5])

                    district1 = next((k for k, v in district1_mapping.items() if v == district1_code), None)
                    if not district1:
                        return {"error": "Invalid district1 code"}

                    district2 = next(
                        (k for k, v in district2_mapping.items() 
                        if v == district2_code and hierarchy[k]["district1"] == district1),
                        None
                    )
                    if not district2:
                        return {"error": "District2 not belong to District1"}

                    district3_list = hierarchy[district2]["district3"]
                    if district3_code < 1 or district3_code > len(district3_list):
                        return {"error": "Invalid district3 index"}

                    district3 = district3_list[district3_code - 1]

                    return {
                        "district1": district1,
                        "district2": district2,
                        "district3": district3,
                        "full_name": f"{district3} ({district2}, {district1})"
                    }
                    
                except Exception as e:
                    return {"error": f"Decoding failed: {str(e)}"}
            ret = decode_district(predicted_district_code)

            st.markdown(f"""
                <div style="
                    padding: 16px;
                    border-radius: 8px;
                    background-color: #f0f9f0;
                    border-left: 5px solid #2ecc71;
                    margin: 10px 0;
                ">
                    <h3 style="color: #27ae60; margin-top: 0;">✨ Recommended Location</h3>
                    <h6>tip: place will be displayed from big area to small </h6>
                    <h5 style="margin-bottom: 5px;"><b>🏙️ Region:</b> {ret['district1']}</h5>
                    <h5><b>🏘️ District:</b> {ret['district2']}</h5>
                    <h5><b>📍 Detailed place:</b> {ret['district3']}</h5>
                </div>
                """, unsafe_allow_html=True)
            
            #test
            # with st.expander("View input features (for testing)"):
            #     st.json(features)

        except Exception as e:
            st.error(f"Error getting recommendations: {e}")



def main():
    st.sidebar.title("Navigation")

    page = st.sidebar.radio("Go to", ["Statistics Analysis", "Price Prediction", "Recommended location"])
    
    if page == "Statistics Analysis":
        show_statistics_analysis()
    elif page == "Price Prediction":
        show_predict_page()
    elif page == "Recommended location":
        show_recommended_location()

if __name__ == "__main__":
    main()
