from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

model = joblib.load('final_model.pkl')
vectorizer = joblib.load('final_vectorizer.pkl')
#pickle делает это в 4 строки

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    category: str
    probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: TextRequest):
    X = vectorizer.transform([request.text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    max_proba = max(proba)
    return {"category": pred, "probability": round(float(max_proba) * 100, 1)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)