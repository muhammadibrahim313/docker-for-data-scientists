# train_model.py
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

# Save
artifact = {
    "model": model,
    "feature_names": iris.feature_names,
    "target_names": iris.target_names.tolist()
}
joblib.dump(artifact, "model.pkl")
print("Saved: model.pkl")