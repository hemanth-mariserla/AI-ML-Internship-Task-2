import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# -------------------------------
# Step 1: Basic Data Summary
# -------------------------------
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include='all'))

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# Step 2: Histograms
# -------------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

df[numeric_cols].hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.suptitle("Histograms of Numeric Columns", y=1.02)
plt.show()

# -------------------------------
# Step 3: Boxplots
# -------------------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# -------------------------------
# Step 4: Correlation Heatmap
# -------------------------------
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# -------------------------------
# Step 5: Pairplot
# -------------------------------
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']], hue='Survived')
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()

# -------------------------------
# Step 6: Feature-level Inferences
# -------------------------------

# Survival by Sex
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Sex")
plt.show()

# Survival by Pclass
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival Count by Passenger Class")
plt.show()

# Age distribution by Survival
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='Age', hue='Survived', fill=True)
plt.title("Age Distribution by Survival")
plt.show()

# -------------------------------
# Step 7: Interactive Plot with Plotly
# -------------------------------
fig = px.scatter(df, x='Age', y='Fare', color='Survived',
                 hover_data=['Name', 'Sex', 'Pclass'],
                 title="Age vs Fare (Colored by Survival)")
fig.show()
