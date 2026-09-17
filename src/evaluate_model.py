import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)


# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "landmarks" / "landmarks.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "bsl_sign_language_rf.pkl"


# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["label"])
y = df["label"]


# -----------------------------
# Create the same test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load(MODEL_PATH)


# -----------------------------
# Make predictions
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(f"\nTest samples: {len(X_test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# -----------------------------
# Classification report
# -----------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------
# Confusion matrix
# -----------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    xticks_rotation="vertical"
)

plt.title("BSL Fingerspelling Confusion Matrix")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "confusion_matrix.png",
    dpi=300
)

plt.show()


# -----------------------------
# Class distribution
# -----------------------------

class_counts = y.value_counts().sort_index()

print("\nClass Distribution:")
print(class_counts)

class_counts.plot(
    kind="bar",
    figsize=(12, 5)
)

plt.title("BSL Dataset Class Distribution")
plt.xlabel("Letter")
plt.ylabel("Number of Samples")
plt.tight_layout()

plt.savefig(
    PROJECT_ROOT / "class_distribution.png",
    dpi=300
)

plt.show()
from sklearn.metrics import confusion_matrix

print("\nPredicted Class Distribution:")
print(pd.Series(y_pred).value_counts().sort_index())

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=sorted(y.unique())
    )
)
