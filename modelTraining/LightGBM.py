import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

class SafeTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.maps = {}
        self.global_mean = None
    
    def fit(self, X, y):
        self.global_mean = np.mean(y)
        for col in self.cols:
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            stats['smooth'] = (stats['mean'] * stats['count'] + self.global_mean * self.smoothing) / (stats['count'] + self.smoothing)
            self.maps[col] = stats['smooth'].to_dict()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.cols:
            encoded_col = X_copy[col].map(self.maps[col])
            encoded_col = encoded_col.fillna(self.global_mean)
            X_copy[col+'_encoded'] = encoded_col
        return X_copy

# load data
df = pd.read_excel("data/hongkong_shop_rent.xlsx")

# data prepreparation
df = df.drop(columns=['name'])
df = df.assign(floor_encoded=df['floor'].map({'G/F': 0, 'Low': 1, 'Mid': 2, 'High': 3}))

# partition
X = df.drop(columns=['avg_price_monthly'])
y = df['avg_price_monthly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# target encode
encoder = SafeTargetEncoder(cols=['district1', 'district2', 'district3'], smoothing=5.0)
encoder.fit(X_train, y_train)
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)

# combine features
onehot_cols = ['usage', 'air_con']
numeric_cols = ['floor_encoded', 'transportation_time(min)', 'store_age', 'vendor_points', 'AQI']
district_encoded = [col+'_encoded' for col in ['district1', 'district2', 'district3']]

X_train_final = pd.concat([
    X_train_encoded[numeric_cols + district_encoded],
    pd.get_dummies(X_train_encoded[onehot_cols], drop_first=True)
], axis=1)

X_test_final = pd.concat([
    X_test_encoded[numeric_cols + district_encoded],
    pd.get_dummies(X_test_encoded[onehot_cols], drop_first=True)
], axis=1)


missing_cols = set(X_train_final.columns) - set(X_test_final.columns)
for col in missing_cols:
    X_test_final[col] = 0
X_test_final = X_test_final[X_train_final.columns]

# model training
train_data = lgb.Dataset(X_train_final, label=y_train)
test_data = lgb.Dataset(X_test_final, label=y_test, reference=train_data)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}

model = lgb.train(params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, test_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(50)
                ])

# model evaluation
y_pred = model.predict(X_test_final)
print(f"\nMeaseres Performance:")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# features importance
feature_importance = pd.DataFrame({
    'feature': X_train_final.columns,
    'importance': model.feature_importance()
}).sort_values('importance', ascending=False)

print("\nTop 20 feature importance:")
print(feature_importance.head(20))

plt.figure(figsize=(12, 8))
lgb.plot_importance(model, max_num_features=20)
plt.title("Feature Importance")
plt.show()
