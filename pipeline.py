import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# EXTRACT
# -------------------------------

print("Loading Dataset...")

df = pd.read_csv("tested.csv")

print("\nOriginal Dataset:")
print(df.head())

# -------------------------------
# TRANSFORM
# -------------------------------

print("\nCleaning Data...")

print(df.columns)

# Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Remove Duplicates
df.drop_duplicates(inplace=True)

# Encode Categorical Columns
label_encoder = LabelEncoder()

categorical_cols = ['Sex', 'Embarked']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Feature Scaling
scaler = StandardScaler()

numeric_cols = ['Age', 'Fare']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -------------------------------
# LOAD
# -------------------------------

output_file = "cleaned_csv_file.csv"
df.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved as {output_file}")

print("\nProcessed Dataset:")
print(df.head())