"""Binary tabular classification with F1-weighted probability voting."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True, help='Labeled CSV; target in final column')
    parser.add_argument('--test', type=Path, required=True, help='Unlabeled CSV with matching features')
    parser.add_argument('--output', type=Path, default=Path('result.csv'))
    parser.add_argument('--imputer', choices=('missforest', 'knn', 'median', 'mean', 'interpolate'), default='missforest')
    return parser.parse_args()


def prune_missing(df: pd.DataFrame, steps: int = 40) -> pd.DataFrame:
    """Gradually remove columns and rows above the original missingness limits."""
    result = df.copy()
    for column_limit, row_limit in zip(np.linspace(1, .10, steps), np.linspace(1, .15, steps)):
        result = result.loc[:, result.isna().mean() < column_limit]
        result = result.loc[result.isna().mean(axis=1) < row_limit].reset_index(drop=True)
    return result


def fill_missing(df: pd.DataFrame, method: str) -> pd.DataFrame:
    result = df.copy()
    columns = result.columns[:-1]
    if method == 'missforest':
        # missingpy imports the former sklearn.neighbors.base module.
        import sklearn.neighbors._base
        sys.modules.setdefault('sklearn.neighbors.base', sklearn.neighbors._base)
        from missingpy import MissForest
        imputer = MissForest(max_iter=10, decreasing=False, n_estimators=100,
                             criterion=('mse', 'gini'), max_features='auto',
                             n_jobs=-1, random_state=1)
        result.loc[:, columns] = imputer.fit_transform(result.loc[:, columns])
    elif method == 'knn':
        result.loc[:, columns] = KNNImputer(n_neighbors=10).fit_transform(result.loc[:, columns])
    elif method in ('median', 'mean'):
        statistic = result.loc[:, columns].median() if method == 'median' else result.loc[:, columns].mean()
        result.loc[:, columns] = result.loc[:, columns].fillna(statistic)
    else:
        result.loc[:, columns] = result.loc[:, columns].interpolate()
    return result


def prepare_data(train: pd.DataFrame, test: pd.DataFrame, imputer: str):
    if list(train.columns[:-1]) != list(test.columns):
        raise ValueError('Test columns must match training feature columns in order')
    target = train.columns[-1]
    cleaned = prune_missing(train.drop_duplicates().reset_index(drop=True))
    if target not in cleaned:
        raise ValueError('Target column was removed during missing-value filtering')
    filled = fill_missing(cleaned, imputer)
    # Preserve the original combined feature engineering and scaling sequence.
    combined = pd.concat([filled, test.reindex(columns=filled.columns)], ignore_index=True)
    features = combined.columns[:-1]
    values = combined.loc[:, features]
    mean = values.mean()
    std = values.std(ddof=0)
    combined.loc[:, features] = values.clip(lower=mean - 10 * std, upper=mean + 10 * std, axis=1)
    retained = combined.loc[:, features].var() > .005
    features = features[retained.to_numpy()]
    correlation = combined.loc[:, features].corr().abs()
    rejected = [features[i] for i in range(len(features)) if (correlation.iloc[:i, i] > .9).any()]
    features = features.drop(rejected)
    scaled = MinMaxScaler().fit_transform(combined.loc[:, features])
    split = len(filled)
    return scaled[:split], filled[target].to_numpy(), scaled[split:]


def models():
    return {
        'random_forest': RandomForestClassifier(class_weight='balanced', criterion='entropy',
            max_depth=None, max_features=.5, n_estimators=300, random_state=42),
        'knn': KNeighborsClassifier(metric='manhattan', n_neighbors=9, weights='uniform'),
        'naive_bayes': MultinomialNB(alpha=.01),
    }


def scores(y_true, probabilities):
    predictions = (probabilities >= .5).astype(int)
    return metrics.accuracy_score(y_true, predictions), metrics.f1_score(y_true, predictions, zero_division=0)


def evaluate(x: np.ndarray, y: np.ndarray):
    folds = KFold(n_splits=5, shuffle=False)
    weights = {}
    for name, model in models().items():
        fold_scores = []
        for train_idx, valid_idx in folds.split(x):
            model.fit(x[train_idx], y[train_idx])
            fold_scores.append(scores(y[valid_idx], model.predict_proba(x[valid_idx])[:, 1]))
        weights[name] = round(float(np.mean(fold_scores, axis=0)[1]), 3)
    if sum(weights.values()) == 0:
        raise ValueError('All model F1 weights are zero')

    ensemble_scores = []
    for train_idx, valid_idx in folds.split(x):
        probabilities = []
        for model in models().values():
            model.fit(x[train_idx], y[train_idx])
            probabilities.append(model.predict_proba(x[valid_idx])[:, 1])
        weighted = np.average(probabilities, axis=0, weights=list(weights.values()))
        ensemble_scores.append(scores(y[valid_idx], weighted))
    return weights, np.mean(ensemble_scores, axis=0)


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False)
    x, y, x_test = prepare_data(train, test, args.imputer)
    weights, (accuracy, f1) = evaluate(x, y)
    probabilities = []
    for model in models().values():
        model.fit(x, y)
        probabilities.append(model.predict_proba(x_test)[:, 1])
    predictions = (np.average(probabilities, axis=0, weights=list(weights.values())) > .5).astype(int)
    output = pd.DataFrame({'prediction': predictions, 'f1': np.nan})
    output.loc[len(output)] = [f'{accuracy:.3f}', f'{f1:.3f}']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, header=False, na_rep='')
    print(f'Saved {len(predictions)} predictions to {args.output}; CV accuracy={accuracy:.3f}, F1={f1:.3f}')


if __name__ == '__main__':
    main()
