from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  # This was your missing import
import matplotlib.pyplot as plt

file_path = 'data/stores_rent.xlsx'
data = pd.read_excel(file_path)

# Floor encoding
floor_mapping = {
    'G/F ': 1,
    'Low': 2,
    'Mid': 3,
    'High': 4
}
data['floor_encoded'] = data['floor'].map(floor_mapping)

# One-hot encoding: for building type (non-ordinal categorical variables)
building_type_encoder = OneHotEncoder(sparse_output=False)
building_type_encoded = building_type_encoder.fit_transform(data[['usage']])

# One-hot encoding for air conditioner
air_con_type_encoder = OneHotEncoder(sparse_output=False)
air_con_type_encoded = air_con_type_encoder.fit_transform(data[['air_con']])

# Convert one-hot encoded results to DataFrame and merge
building_type_encoded_df = pd.DataFrame(building_type_encoded,
                                      columns=building_type_encoder.categories_[0])
data = pd.concat([data, building_type_encoded_df], axis=1)
# Convert air conditioner one-hot encoded results to DataFrame and merge
air_con_type_encoded_df = pd.DataFrame(air_con_type_encoded,
                                    columns=air_con_type_encoder.categories_[0])
data = pd.concat([data, air_con_type_encoded_df], axis=1)


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

# Import required libraries
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import numpy as np

# 1. Define features and target variable
features = [
    'floor_encoded', 'AQI', 'vendor_points', 'store_age',
    'transportation_time(min)', 'area(foot)', 'Industrial', 'Office', 'avg_price_monthly',
    'Cen_air_con', 'Chilled_Water_sys','Hybrid_air_con', 'Indi_air_con', 'Split_air_con', 'Others',
]
target = 'district_code'
# Prepare features and target variable
X = data[features]
y = data[target]

# 3. Handle missing values
# Fill numerical features with mean
num_imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(num_imputer.fit_transform(X), columns=X.columns)

# Fill categorical target with mode
mode_value = y.mode()[0]
y = y.fillna(mode_value)
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create decision tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Encapsulation
import joblib
import os
current_directory = os.getcwd()
directory_contents = os.listdir(current_directory)
print(current_directory, directory_contents)
model_save_path = 'models/decision_tree.pkl'
try:
    joblib.dump(dt_model, model_save_path)
    print(f"\nModel saved to: {model_save_path}")
except Exception as e:
    print(f"Error saving model: {e}")

# Evaluate model
y_pred = dt_model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")


def decode_district(code):
    try:
        code_str = str(code).zfill(5)
        district1_code = int(code_str[0])
        district2_code = int(code_str[1:3])
        district3_code = int(code_str[3:5])

        # 1. Decode district1
        district1 = next((k for k, v in district1_mapping.items() if v == district1_code), None)
        if not district1:
            return {"error": "Invalid district1 code"}

        # 2. Decode district2 and verify it belongs to district1
        district2 = next(
            (k for k, v in district2_mapping.items() 
            if v == district2_code and hierarchy[k]["district1"] == district1),
            None
        )
        if not district2:
            return {"error": "District2 does not belong to District1"}

        # 3. Decode district3
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


def recommend_location(input_features):
    """
    Recommend optimal location based on input features

    Parameters:
    input_features (dict): Dictionary containing required model features

    Returns:
    dict: Dictionary containing recommended areas and probabilities
    """
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_features])

    # Ensure all features exist
    for feature in features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Or use appropriate default value

    # Predict probabilities
    proba = dt_model.predict_proba(input_df)[0]

    # Get top 3 most likely areas
    top3_indices = proba.argsort()[-3:][::-1]
    top3_codes = dt_model.classes_[top3_indices]
    top3_probs = proba[top3_indices]

    recommendations = []
    for code, prob in zip(top3_codes, top3_probs):
        decoded = decode_district(code)
        decoded['probability'] = f"{prob * 100:.1f}%"
        recommendations.append(decoded)

    return recommendations


# Visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 12))
plot_tree(dt_model,
          feature_names=features,
          class_names=[str(code) for code in dt_model.classes_],
          filled=True,
          rounded=True,
          max_depth=3)  # Limit depth for readability
plt.title("Site selection decision tree (first 3 layers)")
plt.show()
# Get feature importance
feature_importance = dt_model.feature_importances_
sorted_idx = feature_importance.argsort()

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Importance ranking of site selection decision features")
plt.tight_layout()
plt.show()