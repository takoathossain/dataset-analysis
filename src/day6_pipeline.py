
import os, pandas as pd, json, joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Setup Paths
base_path = r"E:/SUST/4-2/Asmicore/ML/Day 1"
data_path = os.path.join(base_path, "data/dataset.csv")

# 1. Load Data
df = pd.read_csv(data_path)

# 2. Simple Preprocessing (Inside the script)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. Define Features and Target
features = ['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']
X = df[features]
y = df['2urvived']

# 4. Build the Pipeline
# This bundles the scaler and the model into one unit
pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('clf', LogisticRegression())
])

# 5. Fit the entire pipeline
pipeline.fit(X, y)

# 6. Export the Model
model_dir = os.path.join(base_path, "models")
os.makedirs(model_dir, exist_ok=True)
model_file = os.path.join(model_dir, "takoat_titanic_model.pkl")
joblib.dump(pipeline, model_file)

# 7. Generate Final Output JSON
output_data = {
    "day": 6,
    "pipeline_steps": ["StandardScaler", "LogisticRegression"],
    "model_path": "models/takoat_titanic_model.pkl",
    "status": "Complete"
}

with open(os.path.join(base_path, "outputs/takoat_week1day6_pipeline.json"), 'w') as f:
    json.dump(output_data, f, indent=4)

print("Day 6: Pipeline created, Model exported, and JSON generated.")
