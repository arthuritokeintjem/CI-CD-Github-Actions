import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

MODEL_PATH = Path("model/model.pkl")
clf = joblib.load(MODEL_PATH)

app = FastAPI(title="Iris Classifier API")

class IrisFeatures(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Iris API up"}

@app.post("/predict")
def predict(payload: IrisFeatures):
    x = np.array(payload.features).reshape(1, -1)
    pred = int(clf.predict(x)[0])
    return {"prediction": pred}