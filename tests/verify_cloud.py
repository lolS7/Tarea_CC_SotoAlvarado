"""Registra evidencia HTTP del servicio público para la entrega."""
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = 'https://canciones-api-412738758691.southamerica-west1.run.app'


def main():
    payload = json.loads((ROOT / 'examples/cancion.json').read_text(encoding='utf-8'))
    checks = []
    for path, body in [('/health', None), ('/docs', None),
                       ('/openapi.json', None), ('/prediccion/', payload)]:
        request = Request(BASE_URL + path,
                          data=None if body is None else json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=120) as response:
            status = response.status
            content_type = response.headers.get('Content-Type', '')
            text = response.read().decode('utf-8')
        assert status == 200, (path, status)
        result = {'method': 'GET' if body is None else 'POST',
                  'url': BASE_URL + path, 'status': status, 'content_type': content_type}
        if path == '/docs':
            assert 'SwaggerUIBundle' in text and '/openapi.json' in text
            result['swagger_html_present'] = True
        else:
            decoded = json.loads(text)
            if path == '/health':
                assert decoded == {'status': 'ok', 'model_loaded': True}
                result['response'] = decoded
            elif path == '/openapi.json':
                assert '/prediccion/' in decoded['paths']
                result['prediction_endpoint_documented'] = True
            else:
                assert isinstance(decoded['EsExito'], bool)
                assert 0 <= decoded['ProbabilidadExito'] <= 1
                result.update(request=body, response=decoded)
        checks.append(result)
    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'base_url': BASE_URL, 'authentication': 'none',
              'passed': len(checks), 'checks': checks}
    output = ROOT / 'docs/cloud_test_results.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
