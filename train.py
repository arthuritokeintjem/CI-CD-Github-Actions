import json, joblib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    ConfusionMatrixDisplay
)

# 1. Load dataset & split
X, y = load_iris(return_X_y=True)  # dataset iris bawaan scikit-learn
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2. Train model
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, ytr)

# 3. Evaluasi
y_pred = clf.predict(Xte)
acc = accuracy_score(yte, y_pred)
rmse = np.sqrt(mean_squared_error(yte, y_pred))
# MAPE di sini agak tricky karena y berupa label kategori (0,1,2).
# Secara konsep MAPE cocoknya buat data numerik/continuous.
# Untuk demo, tetap dihitung tapi hati-hati interpretasi.
mape = np.mean(np.abs((yte - y_pred) / (yte + 1e-9))) * 100  

# 4. Simpan artifacts
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(clf, "artifacts/model.pkl")
with open("artifacts/metrics.json", "w") as f:
    json.dump({
        "accuracy": acc,
        "rmse": rmse,
        "mape": mape
    }, f, indent=2)

# 5. Confusion matrix plot
ConfusionMatrixDisplay.from_estimator(clf, Xte, yte)
plt.title("Confusion Matrix")
plt.savefig("artifacts/confusion_matrix.png")

# 6. Gate performa sederhana
if acc < 0.90:
    raise SystemExit(f"Accuracy too low: {acc:.3f}")