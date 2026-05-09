
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Setup Paths
base_path = r"E:/SUST/4-2/Asmicore/ML/Day 1"
data_path = os.path.join(base_path, "data", "processed_dataset.csv")

# 2. Load Processed Data
df = pd.read_csv(data_path)

# 3. Define Features (X) and Target (y)
# Selecting the 7 key features identified in your Day 1 report
features = ['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']
X = df[features]
y = df['2urvived']

# 4. Train-Test Split (80% Train, 20% Test)
# We split the data so we can evaluate the model on 'unseen' data later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Generate Predictions
predictions = model.predict(X_test)

# 7. Generate JSON Output
model_output = {
    "day": 4,
    "model_type": "Logistic Regression",
    "train_test_split": "80/20",
    "sample_predictions": predictions[:10].tolist(),
    "features_used": features
}

output_json_path = os.path.join(base_path, "outputs", "takoat_week1day4_models.json")
with open(output_json_path, 'w') as f:
    json.dump(model_output, f, indent=4)

print("Day 4: Model Training and Predictions Complete.")
