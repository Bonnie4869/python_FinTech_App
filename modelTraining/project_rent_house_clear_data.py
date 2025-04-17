import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def train_and_predict_model(file_path, features, target):
    """
    Train a gradient lift regression model and make predictions.

    Parameters:
    - file_path: indicates the path of the data file
    - features: indicates the list of features used for training
    - target: indicates the predicted target variable

    Back:
    - model: The trained model
    -r2: indicates the R² score of the model
    - rmse: indicates the RMSE value of the model
    - feature_importance: indicates the importance of features
    - y_test: specifies the true value of the test set
    - y_pred: predicts the value of the model
    """

    # 1. import the dataset
    file_path = 'D:/HKBU/course/python/project/zxw/data mining/try/model2/data/stores_rent.xlsx'
    data = pd.read_excel(file_path)

    # 2. Floor code
    floor_mapping = {
        'G/F ': 1,
        'Low': 2,
        'Mid': 3,
        'High': 4
    }
    data['floor_encoded'] = data['floor'].map(floor_mapping)

    # 3. One-hot encoding: 
    # One-hot code for building type (non-ordinal categorical variables)
    building_type_encoder = OneHotEncoder(sparse_output=False)
    building_type_encoded = building_type_encoder.fit_transform(data[['usage']])
    #One-hot code for air conditioner
    air_con_type_encoder = OneHotEncoder(sparse_output=False)
    air_con_type_encoded = air_con_type_encoder.fit_transform(data[['air_con']])

    # convert one-hot encoded results to DataFrame and merge
    building_type_encoded_df = pd.DataFrame(building_type_encoded,
                                        columns=building_type_encoder.categories_[0])
    data = pd.concat([data, building_type_encoded_df], axis=1)
    air_con_type_encoded_df = pd.DataFrame(air_con_type_encoded,
                                        columns=air_con_type_encoder.categories_[0])
    data = pd.concat([data, air_con_type_encoded_df], axis=1)


    # 4. District encoding mapping
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



    # Create district3_mapping using different variable names
    district3_code_mapping = {}
    for district2, district_data in hierarchy.items():  # Renamed to district_data to avoid conflict
        for i, district3 in enumerate(district_data["district3"], 1):
            code = district1_mapping[district_data["district1"]] * 10000 + district2_mapping[district2] * 100 + i
            district3_code_mapping[district3] = code

    # Check if district3 column exists
    if 'district3' in data.columns:
        data['district_code'] = data['district3'].map(district3_code_mapping)
    else:
        print("Warning: 'district3' column does not exist in the data")

    # 5. features
    features = [
        'floor_encoded', 'AQI', 'vendor_points', 'store_age',
        'transportation_time(min)', 'area(foot)', 'Industrial', 'Office', 'district_code',
        'Cen_air_con', 'Chilled_Water_sys','Hybrid_air_con', 'Indi_air_con', 'Split_air_con', 'Others',
    ]
    target = 'avg_price_monthly'

    # 6. check missing value
    print("missing value:")
    print(data[features + [target]].isnull().sum())

    # 7. partition
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 8. fill in missing value
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 9. model training
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    #save model for application
    import joblib
    import os
    current_directory = os.getcwd()
    directory_contents = os.listdir(current_directory)
    print(current_directory, directory_contents)
    model_save_path = 'models/gradient_boosting_model.pkl'
    try:
        joblib.dump(gb_model, model_save_path)
        print(f"\nmodel is save to: {model_save_path}")
    except Exception as e:
        print(f"error: {e}")



    # 10. model evaluation
    y_pred = gb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\nR²: {r2_score(y_test, y_pred):.3f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    # 11. feature importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:\n", importance)

    # 12. visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) 
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual VS Predicted')
    plt.grid(True)

    # R² & RMSE
    plt.text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred):.3f}\nRMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}',
            transform=plt.gca().transAxes, verticalalignment='top')

    plt.show()

    return gb_model, r2, rmse, importance, y_test, y_pred

train_and_predict_model(None,None,None)