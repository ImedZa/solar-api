from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the Solar API. Use /predict for predictions.  NYL Networks !"}
# Charger le modèle
model = joblib.load("solar_model.pkl")

# Définir le format des données d'entrée


class InputData(BaseModel):
    AMBIENT_TEMPERATURE: float
    MODULE_TEMPERATURE: float
    IRRADIATION: float


@app.post("/predict")
async def predict(data: InputData):
    # Convertir les données en DataFrame
    input_df = pd.DataFrame([[
        data.AMBIENT_TEMPERATURE,
        data.MODULE_TEMPERATURE,
        data.IRRADIATION
    ]], columns=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"])

    # Faire la prédiction
    prediction = model.predict(input_df)

    # Retourner les résultats
    return {
        "DC_POWER": float(prediction[0, 0]),
        "AC_POWER": float(prediction[0, 1])
    }
