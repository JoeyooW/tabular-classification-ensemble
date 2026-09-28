# Tabular Binary Classification with F1-Weighted Voting

A reproducible pipeline for binary classification of numeric tabular data. It cleans missing data, engineers features, evaluates three classifiers by five-fold cross-validation, and combines their predicted probabilities using their cross-validated F1 scores.

## Inputs and outputs

The training CSV must have numeric features followed by a final binary target column (`0` or `1`). The test CSV must contain the same feature columns in the same order, without the target. The output CSV has no header: one prediction (`0` or `1`) per test row, an empty second field, then a final row with cross-validation accuracy and F1 rounded to three decimals. For a test set of 817 rows, the output has 818 rows.

Training data is not included. Supply it locally and avoid committing private data. The test data is likewise kept out of this repository.

## Setup

Python 3.10 is recommended. The original environment used NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.0.2, SciPy 1.13.1 and missingpy 0.2.0 on macOS 11.5.2. `missingpy` is old and may require an older compatible scikit-learn installation. In a fresh environment, `--imputer median` or `--imputer knn` avoids that optional dependency but changes the method and predictions.

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy scikit-learn
pip install missingpy  # only for default MissForest mode, subject to version compatibility
python main.py --train path/to/training.csv --test path/to/test_data.csv --output result.csv
```

For a dependency-light run: `python main.py --train path/to/training.csv --test path/to/test_data.csv --imputer median`.

## Method

1. Remove duplicate training rows, then iteratively filter columns above 10% missingness and rows above 15% missingness over 40 steps.
2. Impute training features with MissForest (10 iterations, 100 trees, seed 1). KNN, median, mean and interpolation are optional alternatives.
3. Combine processed training rows with the test rows for feature engineering. Clip values to the mean ± 10 population standard deviations; remove features with variance at most 0.005 and later features with absolute Pearson correlation above 0.9 to an earlier feature. Apply MinMax scaling.
4. Fit a random forest (300 trees, entropy, balanced classes, max_features=0.5, seed 42), KNN (9 neighbors, Manhattan distance), and Multinomial Naive Bayes (alpha=0.01).
5. Use unshuffled five-fold cross-validation and threshold 0.5. Round each model's mean F1 to three decimals and use these values as ensemble weights. Report mean fold accuracy and F1; refit all models on the training rows and predict the test set using weighted probabilities, with a strict `> 0.5` final threshold.

The hyperparameters were selected during earlier experiments and are fixed here. The historical notebook/report describes a grid search, but that search is not executed by this script. The original voting evaluation accidentally reused random forest probabilities three times; this implementation evaluates the actual three-model ensemble, so its evaluation score can differ from the historical report. Feature processing and ensemble weights are computed ahead of fold evaluation, as in the source script; reported cross-validation scores are therefore potentially optimistic. For an unbiased estimate, place all preprocessing and weight estimation inside the validation folds.

## Repository layout

- `main.py`: end-to-end pipeline and command-line entry point.
- `.gitignore`: excludes data, output and local environments.
