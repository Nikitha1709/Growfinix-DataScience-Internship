import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("=================================")
print("TITANIC SURVIVAL ANALYSIS")
print("=================================")

# Load Titanic Dataset
df = sns.load_dataset("titanic")

# Dataset Information
print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Data Cleaning
# -----------------------------

df['age'] = df['age'].fillna(df['age'].median())

df['embarked'] = df['embarked'].fillna(
    df['embarked'].mode()[0]
)

# Drop deck column because of many missing values
if 'deck' in df.columns:
    df.drop('deck', axis=1, inplace=True)

# -----------------------------
# Encoding
# -----------------------------

df['sex'] = df['sex'].map({
    'male': 0,
    'female': 1
})

df['embarked'] = df['embarked'].map({
    'S': 0,
    'C': 1,
    'Q': 2
})

# -----------------------------
# Age Groups
# -----------------------------

df['Age_Group'] = pd.cut(
    df['age'],
    bins=[0, 12, 19, 35, 60, 100],
    labels=[
        'Child',
        'Teen',
        'Adult',
        'Middle Age',
        'Senior'
    ]
)

# -----------------------------
# Survival by Gender
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(
    x='sex',
    hue='survived',
    data=df
)
plt.title('Survival by Gender')
plt.savefig('survival_gender.png')
plt.close()

# -----------------------------
# Survival by Class
# -----------------------------

plt.figure(figsize=(6,4))
sns.countplot(
    x='pclass',
    hue='survived',
    data=df
)
plt.title('Survival by Class')
plt.savefig('survival_class.png')
plt.close()

# -----------------------------
# Survival by Age Group
# -----------------------------

plt.figure(figsize=(8,4))
sns.countplot(
    x='Age_Group',
    hue='survived',
    data=df
)
plt.title('Survival by Age Group')
plt.savefig('survival_agegroup.png')
plt.close()

# -----------------------------
# Family Size Analysis
# -----------------------------

df['Family_Size'] = (
    df['sibsp']
    + df['parch']
    + 1
)

family_analysis = df.groupby(
    'Family_Size'
)['survived'].mean()

print("\nFamily Size Survival Rate:")
print(family_analysis)

# -----------------------------
# Fare Analysis
# -----------------------------

fare_analysis = df.groupby(
    'survived'
)['fare'].mean()

print("\nAverage Fare:")
print(fare_analysis)

# -----------------------------
# GroupBy Analysis
# -----------------------------

group_result = df.groupby(
    'pclass'
)['survived'].mean()

print("\nSurvival Rate by Class:")
print(group_result)

# -----------------------------
# Pivot Table
# -----------------------------

pivot_table = pd.pivot_table(
    df,
    values='survived',
    index='pclass',
    columns='sex',
    aggfunc='mean'
)

print("\nPivot Table:")
print(pivot_table)

# -----------------------------
# Export Cleaned Dataset
# -----------------------------

df.to_csv(
    'cleaned_titanic.csv',
    index=False
)

print("\nCleaned dataset saved.")

# -----------------------------
# Conclusions
# -----------------------------

print("\nCONCLUSIONS")
print("1. Females survived more than males.")
print("2. First class passengers survived more.")
print("3. Higher ticket fare increased survival chances.")
print("4. Small families had better survival rates.")
print("5. Passenger class strongly affected survival.")

print("\nProject Completed Successfully!")
