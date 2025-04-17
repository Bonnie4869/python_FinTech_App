# Step01 Environment preparation and data reading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import folium
from statsmodels.tsa.seasonal import seasonal_decompose
file_path = 'stores_rent.xlsx'
data = pd.read_excel(file_path)





# Step02 Data Basic statistics
description = data.describe(include='all')
print(description)
description.to_excel('Statistics.xlsx')





# Step03 Single variable distribution analysis
# 3.1 Numerical variable distribution
def plot_numeric_dist_hist(col, data = data):
    plt.figure(figsize=(6, 5)) # Create a new graphic with a size of 6x5 inches
    sns.histplot(data[col], kde=True) # Plots a histogram of the specified column using seaborn's histplot function and adds a kernel density estimate (KDE)
    plt.title(f'{col} histogram') # Set the title, displayed as column name and "histogram"
    plt.tight_layout() # Automatically adjusts graph parameters to fill the entire graph area
def plot_numeric_dist_QQ(col, data=data):
    plt.figure(figsize=(6, 5))  # Create a new graphic with a size of 6x5 inches
    stats.probplot(data[col], plot=plt) # Use the function to draw a QQ graph for the specified column
    plt.title(f'{col} QQplot') # Set the title, displayed as column name and "QQplot"
    plt.tight_layout() # Automatically adjusts graph parameters to fill the entire graph area
plot_numeric_dist_hist('area(foot)')
plot_numeric_dist_hist('avg_price_monthly')
plot_numeric_dist_hist('transportation_time(min)')
plot_numeric_dist_hist('store_age')
plot_numeric_dist_hist('vendor_points')
plot_numeric_dist_hist('AQI')
plot_numeric_dist_QQ('area(foot)')
plot_numeric_dist_QQ('avg_price_monthly')
plot_numeric_dist_QQ('transportation_time(min)')
plot_numeric_dist_QQ('store_age')
plot_numeric_dist_QQ('vendor_points')
plot_numeric_dist_QQ('AQI')
plt.show()
# 3.2 Categorical variable distribution
def plot_categorical_dist(col, data=data):
    plt.figure(figsize=(10, 6)) # Set the size of the drawing to 10x6 inches
    sns.countplot(y=col, data=data, order=data[col].value_counts().index) # Plot the frequency distribution
    plt.title(f'{col} distribution')
plot_categorical_dist('floor')
plot_categorical_dist('usage')
plot_categorical_dist('district1')
plot_categorical_dist('district2')
plot_categorical_dist('district3')
plot_categorical_dist('air_con')
# Ignore 'name' list
plt.show()





# Step04 Correlation analysis of rent and features
# 4.1 Scatterplot and Spearman correlation coefficient between a numeric column and rent price
corr_matrix = data.select_dtypes(include=np.number).corr(method='spearman') # Displays the Spearman correlation coefficient between a numeric column and a specific column (in this case, 'avg_price_monthly') in the data frame df
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix[['avg_price_monthly']].sort_values(by='avg_price_monthly', ascending=False),
            annot=True, cmap='coolwarm', vmin=-1, vmax=1) # Extract the columns related to 'avg_price_monthly' from the correlation coefficient matrix and sort them in descending order according to the correlation coefficient with 'avg_price_monthly'.
plt.title('The correlation of rent with other variables')
plt.show()
sns.jointplot(x='area(foot)', y='avg_price_monthly', data=data, kind='reg', height=8)
sns.jointplot(x='transportation_time(min)', y='avg_price_monthly', data=data, kind='reg', height=8)
sns.jointplot(x='store_age', y='avg_price_monthly', data=data, kind='reg', height=8)
sns.jointplot(x='vendor_points', y='avg_price_monthly', data=data, kind='reg', height=8)
sns.jointplot(x='AQI', y='avg_price_monthly', data=data, kind='reg', height=8)
plt.show()
# 4.2 Box plot analysis of category variables
plt.figure(figsize=(12, 6))
sns.boxplot(x='floor', y='avg_price_monthly', data=data)
plt.xticks(rotation=45)
plt.title('Rent distribution for different floors')
plt.figure(figsize=(12, 6))
sns.boxplot(x='usage', y='avg_price_monthly', data=data)
plt.xticks(rotation=45)
plt.title('Rent distribution for different usages')
plt.figure(figsize=(12, 6))
sns.boxplot(x='district1', y='avg_price_monthly', data=data)
plt.xticks(rotation=45)
plt.title('Rent distribution for different district1')
plt.figure(figsize=(12, 6))
sns.boxplot(x='district2', y='avg_price_monthly', data=data)
plt.xticks(rotation=45)
plt.title('Rent distribution for different district2')
plt.figure(figsize=(12, 6))
sns.boxplot(x='district3', y='avg_price_monthly', data=data)
plt.xticks(rotation=45)
plt.title('Rent distribution for different district3')
plt.figure(figsize=(12, 6))
sns.boxplot(x='air_con', y='avg_price_monthly', data=data)
plt.xticks(rotation=45)
plt.title('Rent distribution for different air_con')
plt.show()





