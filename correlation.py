import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kruskal, f_oneway
import warnings
warnings.filterwarnings("ignore")

# 1. Read data
file_path = 'data/hongkong_shop_rent.xlsx'
df = pd.read_excel(file_path)

# 2. View basic information of the data
print("Data overview:")
print(df.head())

# 3. Correlation analysis between numerical variables
num_vars = ['avg_price_monthly', 'transportation_time(min)', 'store_age', 'vendor_points', 'AQI']

# Draw heatmap
sns.heatmap(df[num_vars].corr(), annot=True, cmap='coolwarm')
plt.title('Numerical Correlation Heatmap')
plt.show()

# Pearson correlation coefficient
print("\n[Pearson correlation between numerical variables]")
for var in num_vars:
    if var != 'avg_price_monthly':
        corr, pval = pearsonr(df['avg_price_monthly'], df[var])
        print(f"{var} vs avg_price_monthly: r = {corr:.3f}, p = {pval:.3f}")

# 4. Categorical variable correlation analysis function
def plot_box_and_test(var_name):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=var_name, y='avg_price_monthly', data=df)
    plt.title(f"Rent Distribution by {var_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Kruskal-Wallis test
    groups = [df[df[var_name] == cat]['avg_price_monthly'] for cat in df[var_name].dropna().unique()]
    if len(groups) > 1:
        stat, p = kruskal(*groups)
        print(f"Kruskal-Wallis test for {var_name}: H = {stat:.3f}, p = {p:.3f}")
    else:
        print(f"Cannot perform Kruskal-Wallis test for {var_name} (insufficient number of categories)")

# 5. Perform analysis for each categorical variable
cat_vars = ['floor', 'usage', 'air_con', 'district1', 'district2', 'district3']
for cat in cat_vars:
    print(f"\n[Categorical variable: {cat}]")
    plot_box_and_test(cat)

# 6. Optional: Encode floor and air_con as ordinal values (Ordinal Encoding)
floor_order = {'G/F': 0, 'Low': 1, 'Mid': 2, 'High': 3}
df['floor_encoded'] = df['floor'].map(floor_order)
#print(df['floor'].unique())  # View available floor categories


# Pearson correlation (Ordinal variables)
print("\n[Pearson correlation between encoded ordinal variables and rent]")
for col in ['floor_encoded']:
    corr, pval = pearsonr(df['avg_price_monthly'], df[col])
    print(f"{col} vs avg_price_monthly: r = {corr:.3f}, p = {pval:.3f}")