from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import io

app = FastAPI(title="HealthPredict AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load Models ----------------
try:
    diabetes_model = joblib.load("backend/models/diabetes_model.pkl")
except Exception as e:
    print("Warning: couldn't load diabetes model:", e)
    diabetes_model = None

try:
    liver_model = joblib.load("backend/models/liver_model.pkl")
except Exception as e:
    print("Warning: couldn't load liver model:", e)
    liver_model = None

# ---------------- Schemas ----------------
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    Race: str

class LiverInput(BaseModel):
    Age_of_the_patient: float
    Gender_of_the_patient: int
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkphos_Alkaline_Phosphotase: float
    Sgpt_Alamine_Aminotransferase: float
    Sgot_Aspartate_Aminotransferase: float
    Total_Protiens: float
    ALB_Albumin: float
    AG_Ratio_Albumin_and_Globulin_Ratio: float

# ---------------- Diabetes Prediction ----------------
@app.post("/predict/diabetes")
def predict_diabetes(payload: DiabetesInput):
    if diabetes_model is None:
        raise HTTPException(status_code=503, detail="Diabetes model not loaded.")

    known_races = ["White", "Black", "Hispanic"]
    race = payload.Race if payload.Race in known_races else "Other"

    df = pd.DataFrame([{
        "Pregnancies": payload.Pregnancies,
        "Glucose": payload.Glucose,
        "BloodPressure": payload.BloodPressure,
        "BMI": payload.BMI,
        "DiabetesPedigreeFunction": payload.DiabetesPedigreeFunction,
        "Age": payload.Age,
        "Race": race
    }])

    try:
        pred = diabetes_model.predict(df)
        prob = diabetes_model.predict_proba(df).tolist() if hasattr(diabetes_model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting diabetes: {e}")

    return {"prediction": int(pred[0]), "probability": prob}

# ---------------- Liver Prediction ----------------
@app.post("/predict/liver")
def predict_liver(payload: LiverInput):
    if liver_model is None:
        raise HTTPException(status_code=503, detail="Liver model not loaded.")

    # Use snake_case column names exactly as in your trained model
    df = pd.DataFrame([{
        "Age_of_the_patient": payload.Age_of_the_patient,
        "Gender_of_the_patient": payload.Gender_of_the_patient,
        "Total_Bilirubin": payload.Total_Bilirubin,
        "Direct_Bilirubin": payload.Direct_Bilirubin,
        "Alkphos_Alkaline_Phosphotase": payload.Alkphos_Alkaline_Phosphotase,
        "Sgpt_Alamine_Aminotransferase": payload.Sgpt_Alamine_Aminotransferase,
        "Sgot_Aspartate_Aminotransferase": payload.Sgot_Aspartate_Aminotransferase,
        "Total_Protiens": payload.Total_Protiens,
        "ALB_Albumin": payload.ALB_Albumin,
        "AG_Ratio_Albumin_and_Globulin_Ratio": payload.AG_Ratio_Albumin_and_Globulin_Ratio
    }])

    try:
        pred = liver_model.predict(df)
        prob = liver_model.predict_proba(df).tolist() if hasattr(liver_model, "predict_proba") else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting liver disease: {e}")

    return {"prediction": int(pred[0]), "probability": prob}

# ---------------- CSV Analysis ----------------
@app.post("/analyze-file/")
async def analyze_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Couldn't parse CSV: {e}")

    numeric = df.select_dtypes(include='number')
    analysis = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "describe": numeric.describe().to_dict(),
        "correlation": numeric.corr().round(4).to_dict() if numeric.shape[1] > 0 else {}
    }

    return analysis
