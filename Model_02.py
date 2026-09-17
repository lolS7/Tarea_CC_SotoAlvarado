"""Entrenamiento y serialización del pipeline completo de Model_01."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from Model_01 import ROOT, FEATURES, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'songs_normalize.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'model')
    args = parser.parse_args()
    pipeline, metadata = train(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'model.pkl'
    joblib.dump(pipeline, path, compress=3)
    if path.stat().st_size >= 100_000_000:
        raise ValueError('El artefacto supera 100 MB.')
    restored = joblib.load(path)
    samples = pd.read_csv(args.data)[FEATURES]
    np.testing.assert_allclose(restored.predict_proba(samples),
                               pipeline.predict_proba(samples))
    np.testing.assert_array_equal(restored.predict(samples), pipeline.predict(samples))
    metadata['artifact_bytes'] = path.stat().st_size
    metadata['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / 'metadata.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
