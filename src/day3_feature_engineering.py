
import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler

# 1. Setup Paths
base_path = r"E:/SUST/4-2/Asmicore/ML/Day 1"
input_path = os.path.join(base_path, "data", "dataset.csv")
output_csv_path = os.path.join(base_path, "data", "processed_dataset.csv")
output_json_path = os.path.join(base_path, "outputs", "takoat_week1day3_features.json")

# 2. Load Data
df = pd.read_csv(input_path)

# 3. Imputation (Handling Missing Values)
# Filling Age with Median and Embarked with Mode as per Day 1 plan
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 4. Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 5. Save Processed Dataset
df.to_csv(output_csv_path, index=False)

# 6. Generate JSON Output
features_output = {
    "day": 3,
    "transformations": [
        {"column": "Age", "action": "Median Imputation"},
        {"column": "Embarked", "action": "Mode Imputation"},
        {"column": ["Age", "Fare"], "action": "Standard Scaling"}
    ],
    "status": "Success",
    "processed_file": "processed_dataset.csv"
}

with open(output_json_path, 'w') as f:
    json.dump(features_output, f, indent=4)

print("Day 3: Feature Engineering Complete. Files saved.")
