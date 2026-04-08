from fastapi import FastAPI

app = FastAPI()

@app.get("/api/train")
def train_model():
    # Acá va tu código de predict.py
    return {"ok": True, "action": "train"}

@app.get("/api/predict")
def predict_model():
    # Acá va tu código de predict.py
    return {"ok": True, "action": "predict"}

@app.get("/api/metrics")
def get_metrics():
    # Acá va tu código de metrics.py
    return {"ok": True, "action": "metrics"}
