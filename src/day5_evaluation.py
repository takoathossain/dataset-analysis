
import os, pandas as pd, json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

base_path = r"E:/SUST/4-2/Asmicore/ML/Day 1"
df = pd.read_csv(os.path.join(base_path, "data/processed_dataset.csv"))
X = df[['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked']]
y = df['2urvived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression().fit(X_train, y_train)
report = classification_report(y_test, model.predict(X_test), output_dict=True)

# Find the survival key (could be '1', '1.0', or 1)
s_key = '1' if '1' in report else ('1.0' if '1.0' in report else 1)

output = {
    "day": 5, 
    "accuracy": round(accuracy_score(y_test, model.predict(X_test)), 4),
    "precision": round(report[str(s_key)]['precision'], 4), 
    "f1_score": round(report[str(s_key)]['f1-score'], 4)
}

with open(os.path.join(base_path, "outputs/takoat_week1day5_eval.json"), 'w') as f:
    json.dump(output, f, indent=4)
print("Day 5: Script Created and JSON Generated Successfully.")
