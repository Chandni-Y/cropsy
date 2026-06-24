import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

print("Starting retraining process...")

# Path to dataset
csv_path = os.path.join("Data-processed", "crop_recommendation.csv")
if not os.path.exists(csv_path):
    # Fallback to alternative location
    csv_path = os.path.join("Data", "crop_recommendation.csv")

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# Features and target
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Train Random Forest Classifier
print("Training RandomForestClassifier...")
rf_model = RandomForestClassifier(n_estimators=20, random_state=0)
rf_model.fit(features, target)

# Save the model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "RandomForest.pkl")

print(f"Saving retrained model to: {model_path}")
with open(model_path, "wb") as f:
    pickle.dump(rf_model, f)

print("Retraining completed successfully!")
