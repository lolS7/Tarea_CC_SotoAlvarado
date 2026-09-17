"""Preprocesamiento, selección y entrenamiento reproducible de popularidad."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score,
                             roc_auc_score, precision_score, recall_score)
from sklearn.model_selection import (StratifiedGroupKFold, cross_val_predict,
                                     ParameterGrid, FixedThresholdClassifier)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ['artist', 'duration_ms', 'danceability', 'loudness', 'key', 'genre',
            'energy', 'tempo']
CATEGORICAL = ['artist', 'key']
NUMERIC = ['duration_ms', 'danceability', 'loudness', 'energy', 'tempo']
SEED = 42
POPULARITY_THRESHOLD = 50
DECISION_THRESHOLDS = np.round(np.arange(0.20, 0.81, 0.025), 3)


def build_pipeline(estimator):
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('numeric', numeric, NUMERIC),
            ('categorical', categorical, CATEGORICAL),
            # Indicadores binarios por género; conserva "hip hop" como una unidad.
            # Componentes estándar: el PKL no depende de funciones personalizadas.
            ('genres', CountVectorizer(binary=True, lowercase=True,
                 token_pattern=r'(?u)[^,\s](?:[^,]*[^,\s])?'), 'genre')])),
        ('classifier', estimator)])


def load_data(path):
    raw = pd.read_csv(path)
    missing = set(FEATURES + ['popularity', 'song']) - set(raw.columns)
    if missing:
        raise ValueError(f'Faltan columnas: {sorted(missing)}')
    data = raw.drop_duplicates().reset_index(drop=True)
    if len(data) < 500:
        raise ValueError('Se requieren al menos 500 filas después de deduplicar.')
    if data['popularity'].isna().any() or not data['popularity'].between(0, 100).all():
        raise ValueError('popularity debe contener valores entre 0 y 100 sin nulos.')
    if data[['artist', 'song']].isna().any().any():
        raise ValueError('Se requieren artist y song para separar canciones sin fuga.')
    for col in NUMERIC + ['key']:
        data[col] = pd.to_numeric(data[col], errors='raise')
        if np.isinf(data[col]).any():
            raise ValueError(f'{col} contiene infinitos.')
    if data['genre'].isna().any():
        raise ValueError('genre debe contener texto sin nulos.')
    y = (data['popularity'] > POPULARITY_THRESHOLD).astype(int)
    if y.nunique() != 2:
        raise ValueError('El dataset debe contener ambas clases.')
    groups = pd.factorize(pd.MultiIndex.from_frame(data[['artist', 'song']]))[0]
    return data, y, groups, len(raw)


def metrics(y, pred, prob=None):
    result = {'accuracy': float(accuracy_score(y, pred)),
              'f1_macro': float(f1_score(y, pred, average='macro')),
              'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
              'recall_exito': float(recall_score(y, pred, zero_division=0)),
              'precision_exito': float(precision_score(y, pred, zero_division=0))}
    if prob is not None:
        result['roc_auc'] = float(roc_auc_score(y, prob))
    return result


def candidate_models():
    for params in ParameterGrid({'C': [0.1, 1.0, 10.0],
                                 'class_weight': [None, 'balanced']}):
        yield 'logistic_regression', params, LogisticRegression(
            **params, max_iter=3000, random_state=SEED)
    for params in ParameterGrid({'max_depth': [None, 12],
                                 'min_samples_leaf': [1, 4],
                                 'max_features': ['sqrt', 0.2],
                                 'class_weight': [None, 'balanced']}):
        yield 'random_forest', params, RandomForestClassifier(
            **params, n_estimators=100, random_state=SEED, n_jobs=1)


def train(data_path):
    data_path = Path(data_path)
    data, y, groups, raw_count = load_data(data_path)
    X = data[FEATURES]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    assert not set(groups[train_idx]) & set(groups[test_idx])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X_train, y_train, groups[train_idx]))
    scores, best_rank, best_estimator = [], None, None
    for name, params, estimator in candidate_models():
        print(f'Evaluando {len(scores) + 1}/22: {name} {params}', file=sys.stderr, flush=True)
        # Cada probabilidad procede de un modelo que no vio esa canción.
        oof_prob = cross_val_predict(build_pipeline(estimator), X_train, y_train,
                                     cv=folds, method='predict_proba', n_jobs=1)[:, 1]
        threshold_scores = []
        for threshold in DECISION_THRESHOLDS:
            threshold_scores.append({'threshold': float(threshold),
                                     **metrics(y_train, oof_prob >= threshold)})
        rank = lambda item: (item['accuracy'], item['f1_macro'],
                             item['recall_exito'], -abs(item['threshold'] - 0.5))
        best = max(threshold_scores, key=rank)
        scores.append({'model': name, 'params': params, 'best_threshold': best,
                       'metrics_at_0_5': metrics(y_train, oof_prob >= 0.5, oof_prob),
                       'threshold_search': threshold_scores})
        if best_rank is None or rank(best) > best_rank:
            best_rank, best_estimator = rank(best), estimator
            selected, selected_params, selected_threshold = name, params, best['threshold']
            selected_cv = best
    # El umbral queda dentro del mismo PKL y se aplica también en predict().
    pipeline = build_pipeline(FixedThresholdClassifier(
        best_estimator, threshold=selected_threshold, response_method='predict_proba'))
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(1)]
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    metadata = {
        'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'features': FEATURES, 'categorical_features': CATEGORICAL,
        'target': 'int(popularity > 50)', 'popularity_threshold': POPULARITY_THRESHOLD,
        'decision_threshold': selected_threshold,
        'genre_encoding': 'binary indicators for individual genres, lowercase; hip hop retained',
        'selected_params': selected_params, 'selected_cv': selected_cv,
        'selected_model': selected, 'seed': SEED,
        'dataset_file': data_path.name,
        'dataset_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'rows_original': raw_count, 'rows_after_deduplication': len(data),
        'duplicates_removed': raw_count - len(data),
        'missing_features': X.isna().sum().to_dict(),
        'class_counts': {str(k): int(v) for k, v in y.value_counts().items()},
        'train_rows': len(train_idx), 'test_rows': len(test_idx),
        'split': 'First fold of StratifiedGroupKFold(5, shuffle=True, random_state=42); groups=artist+song',
        'song_group_overlap': 0,
        'selection': '5-fold grouped out-of-fold predictions on training only; maximize accuracy; ties: F1 macro, recall_exito, proximity to 0.5',
        'candidate_count': len(scores),
        'threshold_candidates': DECISION_THRESHOLDS.tolist(),
        'cross_validation': scores,
        'metric': 'accuracy', 'value': float(accuracy_score(y_test, pred)),
        'metrics': metrics(y_test, pred, prob),
        'test_metrics_at_0_5': metrics(y_test, prob >= 0.5, prob),
        'baseline': {'strategy': 'most_frequent',
                     **metrics(y_test, base_pred)},
        'confusion_matrix_labels': [0, 1],
        'confusion_matrix': confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        'classification_report': classification_report(y_test, pred, output_dict=True,
                                                       zero_division=0)}
    return pipeline, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'songs_normalize.csv')
    args = parser.parse_args()
    _, metadata = train(args.data)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
