import json, joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)  # dataset iris bawaan scikit-learn
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=200)
clf.fit(Xtr, ytr)
acc = accuracy_score(yte, clf.predict(Xte))

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(clf, "artifacts/model.pkl")
with open("artifacts/metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f, indent=2)

# Gate performa sederhana:
if acc < 0.90:
    raise SystemExit(f"Accuracy too low: {acc:.3f}")