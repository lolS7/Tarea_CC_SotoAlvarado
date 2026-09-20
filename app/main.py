"""API FastAPI portable; no requiere el CSV para predecir."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.pkl"
METADATA_PATH = ROOT / "model" / "metadata.json"
class Cancion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        allow_inf_nan=False,
    )
    artist: str = Field(min_length=1, max_length=300)
    duration_ms: int = Field(
        gt=0,
        strict=True,
        description="Duración en milisegundos",
    )
    danceability: float = Field(ge=0, le=1)
    loudness: float = Field(description="Sonoridad en dB")
    key: int = Field(ge=0, le=11, strict=True)
    genre: str = Field(min_length=1, max_length=300)
    energy: float = Field(ge=0, le=1)
    tempo: float = Field(gt=0, description="Tempo en pulsaciones por minuto (BPM)")
class Prediccion(BaseModel):
    EsExito: bool
    ProbabilidadExito: float
    model_version: str
    timestamp_utc: str
class ModelInfo(BaseModel):
    model_version: str
    model_type: str
    features: list[str]
    metrics: dict
    timestamp_utc: str
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y los metadatos una sola vez al iniciar la API."""
    app.state.model = joblib.load(MODEL_PATH)
    app.state.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    yield
def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
def _model_version(request: Request) -> str:
    metadata = request.app.state.metadata
    return str(metadata.get("artifact_sha256", "model.pkl"))[:12]
def predict_song(model, song: Cancion, request: Request) -> dict:
    """Transforma una canción y devuelve su predicción."""
    frame = pd.DataFrame([song.model_dump()], columns=model.feature_names_in_)
    probability = model.predict_proba(frame)[0, list(model.classes_).index(1)]
    return {
        "EsExito": bool(model.predict(frame)[0]),
        "ProbabilidadExito": float(probability),
        "model_version": _model_version(request),
        "timestamp_utc": _timestamp_utc(),
    }
app = FastAPI(
    title="Predicción de éxito de canciones",
    version="2.1.0",
    description="Éxito: popularity > 50. Incluye energy y tempo.",
    lifespan=lifespan,
)
@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }
@app.get("/model-info", response_model=ModelInfo)
def model_info(request: Request):
    metadata = request.app.state.metadata
    model = request.app.state.model
    return {
        "model_version": _model_version(request),
        "model_type": type(model).__name__,
        "features": metadata.get("features", []),
        "metrics": metadata.get("metrics", {}),
        "timestamp_utc": _timestamp_utc(),
    }
@app.post("/predict", response_model=Prediccion)
def predict(song: Cancion, request: Request):
    return predict_song(request.app.state.model, song, request)
@app.post("/predict-batch", response_model=list[Prediccion])
def predict_batch(songs: list[Cancion], request: Request):
    return [
        predict_song(request.app.state.model, song, request)
        for song in songs
    ]
@app.post("/prediccion/", response_model=Prediccion)
def prediccion_compatibilidad(song: Cancion, request: Request):
    """Ruta antigua conservada para no romper el uso anterior de la API."""
    return predict_song(request.app.state.model, song, request)
