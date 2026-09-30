from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    TechSupport: str

@app.get("/")
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/predict")
async def predict_churn(data: CustomerData):
    if not model:
        return {"error": "Model not loaded"}
    
    # Create DataFrame from input data
    input_data = pd.DataFrame([{
        "tenure": data.tenure,
        "MonthlyCharges": data.MonthlyCharges,
        "TotalCharges": data.TotalCharges,
        "Contract": data.Contract,
        "InternetService": data.InternetService,
        "TechSupport": data.TechSupport
    }])
    
    # Get probability of churn (class 1)
    probability = model.predict_proba(input_data)[0][1]
    
    return {"churn_probability": float(probability)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
