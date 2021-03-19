
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('model.joblib')

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint

@app.get("/predict")
def predict(acousticness,
    danceability,
    duration_ms,
    energy,
    explicit,
    id,
    instrumentalness
    key,
    liveness,
    loudness,
    mode,
    name,
    release_date,
    speechiness,
    tempo,
    valence,
    artist):
    test={'acousticness':float(params['acousticness']),
        'danceability':float(params['danceability']),
        'duration_ms': int(params['duration_ms']),
        'energy': float(params['energy']),
        'explicit':int(params['explicit']),
        'id': params['id'],
        'instrumentalness': float(params['instrumentalness']),
        'key': int(params['key']),
        'liveness': float(params['liveness']),
        'loudness': float(params['loudness']),
        'mode': int(params['mode']),
        'name': params['name'],
        'release_date':params['release_date'],
        'speechiness':float(params['speechiness']),
        'tempo': float(params['tempo']),
        'valence': float(params['valence']),
        'artist': params['artist']}
    df = pd.DataFrame(test)

    return model.predict(df)


