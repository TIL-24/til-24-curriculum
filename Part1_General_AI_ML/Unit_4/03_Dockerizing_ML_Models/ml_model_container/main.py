from fastapi import FastAPI
from sklearn.linear_model import LinearRegression
import numpy as np

app = FastAPI()

# Simple model training for demonstration
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])
model = LinearRegression().fit(X, y)

@app.get("/predict/")
def predict(value: float):
    prediction = model.predict(np.array([[value]]))
    return {"prediction": prediction.tolist()}
