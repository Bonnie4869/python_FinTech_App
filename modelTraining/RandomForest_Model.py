# Step01 Environment preparation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import r2_score
df = pd.read_excel('stores_rent.xlsx')





# Step02 Coded categorical variable
floor_mapping = {'G/F ': 1,'Low': 2,'Mid': 3,'High': 4}
df['floor'] = df['floor'].map(floor_mapping)
ohe = OneHotEncoder(drop='first')
encoded_features = ohe.fit_transform(df[['usage','air_con', 'district1', 'district2', 'district3']])
encoded_features_dense = encoded_features.toarray()
encoded_df = pd.DataFrame(encoded_features_dense, columns=ohe.get_feature_names_out(['usage','air_con', 'district1', 'district2', 'district3']))
df_processed = pd.concat([df[['floor','area(foot)','transportation_time(min)','store_age','vendor_points','AQI','avg_price_monthly']], encoded_df], axis=1)
df_processed.to_excel('new_data.xlsx')





# Step03 Construct a random forest model
X = df_processed.drop('avg_price_monthly', axis=1)
y = df_processed['avg_price_monthly']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestRegressor(
n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    oob_score=True
)
rf.fit(X_train, y_train)




# Step04 Feature importance analysis and visualization
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
importance.plot(kind='barh', color='teal')
plt.title('Random forest feature importance ranking')
plt.xlabel('Importance score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()





# Step05 Model validation and diagnosis
y_pred = rf.predict(X_test)
print(f"\nModel R²: {r2_score(y_test, y_pred):.3f}")
print(f"OOB score: {rf.oob_score_:.3f}")
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual distribution diagnostic diagram')
plt.xlabel('Predicted value')
plt.ylabel('Residual')
plt.show()





# Step06 Results integration report
print("\n Analysis report")
print(f"1. The three most important characteristics: {importance.index[:3].tolist()}")
print(f"2. The least important three characteristics: {importance.index[-3:].tolist()}")
print("3. Feature importance interpretation suggestions：")
print("   - Importance >0.1: Core influencing factors")
print("   - 0.05< Importance <0.1: minor influencing factor")
print("   - Significance <0.05: possible redundant features")

