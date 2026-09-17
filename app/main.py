"""API FastAPI portable; no requiere el CSV para predecir."""
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]


class Cancion(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True,
                              allow_inf_nan=False)
    artist: str = Field(min_length=1, max_length=300)
    duration_ms: int = Field(gt=0, strict=True, description='Duración en milisegundos')
    danceability: float = Field(ge=0, le=1)
    loudness: float = Field(description='Sonoridad en dB')
    key: int = Field(ge=0, le=11, strict=True)
    genre: str = Field(min_length=1, max_length=300)
    energy: float = Field(ge=0, le=1)
    tempo: float = Field(gt=0, description='Tempo en pulsaciones por minuto (BPM)')


class Prediccion(BaseModel):
    EsExito: bool
    ProbabilidadExito: float


def predict_song(model, song: Cancion):
    frame = pd.DataFrame([song.model_dump()], columns=model.feature_names_in_)
    probability = model.predict_proba(frame)[0, list(model.classes_).index(1)]
    return {'EsExito': bool(model.predict(frame)[0]),
            'ProbabilidadExito': float(probability)}


@asynccontextmanager
async def lifespan(app):
    app.state.model = joblib.load(ROOT / 'model' / 'model.pkl')
    yield


app = FastAPI(title='Predicción de éxito de canciones', version='2.0.0',
              description='Éxito: popularity > 50. Incluye energy y tempo.', lifespan=lifespan)


@app.get('/health')
def health(request: Request):
    return {'status': 'ok', 'model_loaded': hasattr(request.app.state, 'model')}


@app.post('/prediccion/', response_model=Prediccion)
def predecir(song: Cancion, request: Request):
    return predict_song(request.app.state.model, song)
