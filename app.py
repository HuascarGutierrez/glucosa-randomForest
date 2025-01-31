from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir peticiones desde cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los headers
)

modelo = joblib.load('modelo.pkl')
scaler = joblib.load("scaler.pkl")  # Usa el scaler entrenado

@app.get('/predecir/')
async def predecir(Glucose: float, BMI: float):
    nuevo = pd.DataFrame({'Glucose': [Glucose], 'BMI': [BMI]})
    nuevo_scaled = scaler.transform(nuevo)
    prediccion = modelo.predict_proba(nuevo_scaled)

    #rint(nuevo)
    #print(nuevo_scaled)
    #print(prediccion)
    return {'prediccion': prediccion[0][0]}