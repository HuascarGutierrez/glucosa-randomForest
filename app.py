from fastapi import FastAPI
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI()
modelo = joblib.load('modelo.pkl')

@app.get('/predecir/')
async def predecir(Glucose: float, BMI: float):
    scaler = StandardScaler()
    nuevo = np.array([[Glucose, BMI]])
    nuevo_scaled = nuevo
    # nuevo_scaled = scaler.fit_transform(nuevo)
    prediccion = modelo.predict_proba(nuevo_scaled)
    return {'prediccion': prediccion[0][1]}