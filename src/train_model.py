import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "landmarks" / "landmarks.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "bsl_sign_language_rf.pkl"


# Load data
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Separate labels from features
X = df.drop(columns=["label"])
y = df["label"]

print("\nClasses:")
print(sorted(y.unique()))

print("\nClass counts:")
print(y.value_counts().sort_index())


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# Train
print("\nTraining Random Forest...")

model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("TRAINING COMPLETE")
print("==============================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# Detailed results
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to:")
print(MODEL_PATH)
