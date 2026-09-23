from pathlib import Path
from time import perf_counter
import logging

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd


# =========================
# Configuration
# =========================

MODEL_PATH = Path("model/random_forest.joblib")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================
# Chargement du modèle
# =========================

model = joblib.load(MODEL_PATH)


# =========================
# API
# =========================

app = FastAPI(
    title="QCM AI API",
    description="API exposant un modèle Random Forest",
    version="1.0.0"
)


# =========================
# CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Métriques de monitoring
# =========================

metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "errors": 0,
    "total_response_time_ms": 0.0,
    "low_confidence_predictions": 0,
}


# =========================
# Modèles de données
# =========================

class PredictionRequest(BaseModel):
    temps_reponse: float = Field(
        ge=0,
        description="Temps de réponse en secondes"
    )

    nb_erreurs: int = Field(
        ge=0,
        description="Nombre d'erreurs"
    )

    score: float = Field(
        ge=0,
        le=100,
        description="Score entre 0 et 100"
    )


class PredictionResponse(BaseModel):
    satisfait: int
    confidence: float


# =========================
# Routes
# =========================

@app.get("/")
def home():
    return {
        "message": "QCM AI API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):

    start_time = perf_counter()

    metrics["total_requests"] += 1

    try:
        features = pd.DataFrame([{
    "temps_reponse": data.temps_reponse,
    "nb_erreurs": data.nb_erreurs,
    "score": data.score
}])

        # Prédiction        
        prediction = model.predict(features)[0]

        # Probabilités des classes
        probabilities = model.predict_proba(features)[0]

        # Confiance = probabilité maximale
        confidence = float(max(probabilities))

        # Monitoring de la confiance
        if confidence < 0.70:
            metrics["low_confidence_predictions"] += 1

            logger.warning(
                "Prédiction avec faible confiance : %.2f",
                confidence
            )

        metrics["successful_predictions"] += 1

        # Temps de réponse
        response_time = (perf_counter() - start_time) * 1000

        metrics["total_response_time_ms"] += response_time

        logger.info(
            "Prediction=%s | Confidence=%.2f | ResponseTime=%.2f ms",
            prediction,
            confidence,
            response_time
        )

        return {
            "satisfait": int(prediction),
            "confidence": round(confidence, 3)
        }

    except Exception as error:

        metrics["errors"] += 1

        logger.error(
            "Erreur pendant la prédiction : %s",
            error
        )

        raise


@app.get("/metrics")
def get_metrics():

    total = metrics["successful_predictions"]

    if total > 0:
        average_response_time = (
            metrics["total_response_time_ms"] / total
        )

        low_confidence_rate = (
            metrics["low_confidence_predictions"] / total
        ) * 100

    else:
        average_response_time = 0
        low_confidence_rate = 0

    # Vérification du seuil d'alerte
    alert = None

    if low_confidence_rate > 50:
        alert = (
            "ALERTE : plus de 50% des prédictions "
            "ont une faible confiance."
        )

        logger.warning(alert)

    return {
        "total_requests": metrics["total_requests"],
        "successful_predictions": metrics[
            "successful_predictions"
        ],
        "errors": metrics["errors"],
        "average_response_time_ms": round(
            average_response_time,
            2
        ),
        "low_confidence_predictions": metrics[
            "low_confidence_predictions" 
        ],
        "low_confidence_rate_percent": round(
            low_confidence_rate,
            2
        ),
        "alert": alert
    }