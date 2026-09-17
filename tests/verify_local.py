"""Prueba HTTP real en localhost y paridad del PKL, usando la biblioteca estándar."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from app.main import Cancion, predict_song
from Model_01 import FEATURES, load_data


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    payload = json.loads((ROOT / 'examples/cancion.json').read_text(encoding='utf-8'))
    results = []
    model = joblib.load(ROOT / 'model/model.pkl')
    metadata = json.loads((ROOT / 'model/metadata.json').read_text(encoding='utf-8'))
    data, target, _, _ = load_data(ROOT / 'data/songs_normalize.csv')
    assert (data['popularity'] == 50).any()
    assert (data['popularity'] == 51).any()
    assert (target[data['popularity'] == 50] == 0).all()
    assert (target[data['popularity'] == 51] == 1).all()
    results.append({'case': 'popularity_boundary_50_false_51_true', 'passed': True})
    probabilities = model.predict_proba(data[FEATURES])[:, list(model.classes_).index(1)]
    np.testing.assert_array_equal(model.predict(data[FEATURES]),
                                  probabilities >= metadata['decision_threshold'])
    results.append({'case': 'serialized_decision_threshold', 'passed': True,
                    'threshold': metadata['decision_threshold']})
    genres = model.named_steps['preprocessor'].named_transformers_['genres']
    encoded = genres.transform(['rock, pop', ' POP , rock ', 'rock', 'pop']).toarray()
    np.testing.assert_array_equal(encoded[0], encoded[1])
    np.testing.assert_array_equal(encoded[0], np.maximum(encoded[2], encoded[3]))
    assert encoded[0].sum() == 2
    assert 'r&b' in genres.vocabulary_ and 'dance/electronic' in genres.vocabulary_
    assert 'hip hop' in genres.vocabulary_
    results.append({'case': 'combined_genres_order_case_and_individual_indicators', 'passed': True})
    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    (ROOT / 'docs').mkdir(exist_ok=True)
    with (ROOT / 'docs/server.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
             '--port', str(port)], cwd=ROOT, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError('El servidor falló; consultar docs/server.log')
                try:
                    status, health = request('/health')
                    if status == 200:
                        break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.1)
            else:
                raise RuntimeError('Timeout al iniciar el servidor')
            assert health['model_loaded']
            results.append({'case': 'health', 'status': status, 'response': health})
            status, result = request('/prediccion/', payload)
            assert status == 200
            assert result == predict_song(model, Cancion(**payload))
            results.append({'case': 'prediction_matches_serialized_pipeline',
                            'status': status, 'request': payload, 'response': result})
            unknown = {**payload, 'artist': 'Artista nunca observado', 'genre': 'genero nuevo'}
            status, result = request('/prediccion/', unknown)
            assert status == 200 and 0 <= result['ProbabilidadExito'] <= 1
            results.append({'case': 'unknown_categories', 'status': status, 'response': result})
            differing = np.flatnonzero((probabilities >= 0.5) != model.predict(data[FEATURES]))
            if len(differing):
                threshold_payload = data.iloc[int(differing[0])][FEATURES].to_dict()
                status, result = request('/prediccion/', threshold_payload)
                assert status == 200
                assert result['EsExito'] == bool(probabilities[differing[0]] >= metadata['decision_threshold'])
                results.append({'case': 'http_uses_tuned_threshold_instead_of_0_5',
                                'status': status, 'response': result})
            invalid = [dict(payload, duration_ms=-1), dict(payload, key=12),
                       dict(payload, danceability=1.5), dict(payload, artist='  '),
                       dict(payload, popularity=99),
                       {k: v for k, v in payload.items() if k != 'genre'},
                       dict(payload, energy=1.01), dict(payload, energy=-0.1),
                       dict(payload, tempo=0), dict(payload, tempo=-100),
                       {k: v for k, v in payload.items() if k != 'energy'},
                       {k: v for k, v in payload.items() if k != 'tempo'}]
            for i, body in enumerate(invalid):
                status, result = request('/prediccion/', body)
                assert status == 422, (body, result)
                results.append({'case': f'invalid_input_{i + 1}', 'request': body, 'status': status})
            status, schema = request('/openapi.json')
            assert status == 200 and '/prediccion/' in schema['paths']
            results.append({'case': 'openapi_schema', 'status': status})
            assert (ROOT / 'model/model.pkl').stat().st_size < 100_000_000
            report = {'executed_at_utc': datetime.now(timezone.utc).isoformat(),
                      'base_url': url, 'passed': len(results), 'results': results}
            (ROOT / 'docs/local_test_results.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False, indent=2))
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == '__main__':
    main()
