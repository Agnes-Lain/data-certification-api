
from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load('model.joblib')

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint

@app.get("/predict")
def predict(params):

    return model.predict(params)


