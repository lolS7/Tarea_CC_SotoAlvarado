"""Inferencia desde JSON o ingreso interactivo usando el pipeline guardado."""
import argparse
import json

import joblib

from app.main import ROOT, Cancion, predict_song


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=str, help='Ruta a un JSON de entrada')
    args = parser.parse_args()
    if args.json:
        with open(args.json, encoding='utf-8') as file:
            song = Cancion.model_validate(json.load(file))
    else:
        song = Cancion(artist=input('Artista: '),
                       duration_ms=int(input('Duración (ms): ')),
                       danceability=float(input('Danceability (0-1): ')),
                       loudness=float(input('Loudness (dB): ')),
                       key=int(input('Key (0-11): ')), genre=input('Géneros separados por coma: '),
                       energy=float(input('Energy (0-1): ')),
                       tempo=float(input('Tempo (BPM): ')))
    model = joblib.load(ROOT / 'model' / 'model.pkl')
    print(json.dumps(predict_song(model, song), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
