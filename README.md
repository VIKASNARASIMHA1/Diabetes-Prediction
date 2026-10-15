# Diabetes Prediction with Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/notebooks/diabetes_prediction.ipynb)

Predicts whether a patient is diabetic from eight medical measurements, using the **PIMA Indians Diabetes dataset**
(768 patients, 8 features). Three classifiers are compared (linear SVM, logistic regression, random forest) with
cross-validated hyper-parameter search, and the best one is served through a small **Gradio web app**.

> Educational project. **Not medical advice.**

## Results

Held-out test set (154 patients, stratified 80/20 split). The model is selected by **5-fold cross-validated recall on the
training set**; the test set is used once, for reporting.

| Model | CV recall | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| SVM (Linear) | 0.551 | 0.766 | 0.750 | 0.500 | 0.600 |
| Logistic Regression | 0.579 | 0.740 | 0.684 | 0.481 | 0.565 |
| Random Forest (selected) | 0.631 | 0.727 | 0.630 | 0.537 | 0.580 |

Recall is the metric that matters most for screening (a missed diabetic patient is the costly mistake), and it is the
weak point: the models catch only about half of the diabetic patients in the test set. See *Limitations*.

## Methodology note: avoiding data leakage

An earlier version of this project imputed missing values with **per-`Outcome` medians on the full dataset** and fitted the
scaler on all rows before splitting. That leaks the label (and the test rows) into the features. It produced an
attractive-looking 86% test accuracy, and made `Insulin` look like the dominant feature (40% importance) only because ~49% of
its values had been filled in from the answer itself.

This version fixes it: the data is split first, and imputation and scaling live **inside a scikit-learn `Pipeline`**, so
they are learned from training rows only (and re-fitted in every cross-validation fold). Re-running both approaches on five
different random splits with a random forest gave:

| Approach | Mean test accuracy | Mean test recall |
|---|---|---|
| Label-based imputation on all data (leaky) | 86.1% | 81.1% |
| Pipeline after split (this repo) | 75.1% | 58.5% |

The lower numbers are the honest ones: they are what you should expect on new patients.

## Project structure

```
data/diabetes.csv                      PIMA dataset (768 rows)
notebooks/diabetes_prediction.ipynb    full analysis: EDA, models, evaluation (outputs included)
train.py                               trains, evaluates and saves the models (python train.py)
app.py                                 Gradio web app (python app.py)
requirements.txt
```

## Quick start

Python 3.10+ recommended.

```powershell
git clone https://github.com/USERNAME/REPO.git
cd REPO
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python train.py                      # prints the metrics and saves models/diabetes_model.joblib
python app.py                        # opens the web app at http://127.0.0.1:7860
```

To explore the notebook locally: `pip install notebook`, then `jupyter notebook notebooks/diabetes_prediction.ipynb`.

**In Google Colab:** open the notebook with the badge above, then run this in the first cell so the data file is available:

```python
!git clone https://github.com/USERNAME/REPO.git
%cd REPO/notebooks
```

The saved model is not committed to the repository (pickles depend on the scikit-learn version); `app.py` trains one automatically
the first time if it is missing.

## How it works

1. Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are physiologically impossible, so they are treated as missing.
2. Stratified 80/20 train/test split (`random_state=2`).
3. Each model is `Pipeline(median imputer -> StandardScaler -> classifier)`, tuned with `GridSearchCV` (5 folds, scoring = recall).
4. The model with the highest cross-validated recall is selected and evaluated on the test set.
5. The web app takes raw measurements (leave a field empty if unknown), lets you pick the algorithm, and shows the prediction and,
   where the model supports it, the estimated probability.

## Limitations and next steps

- **Small, narrow data:** 768 patients, all women of Pima Indian heritage aged 21+. Results will not generalise to other populations.
- **Low recall (about 50%):** try `class_weight="balanced"`, and tune the decision threshold for recall using the training folds.
- **Overfitting:** the selected random forest scores ~99% on training data vs ~73% on test data; constrain `max_depth` / `min_samples_leaf`.
- **Noisy estimates:** with only 154 test rows, one split is a rough estimate. Use repeated cross-validation and report confidence intervals.
- **Probabilities are uncalibrated;** add calibration before showing percentages to users.
- **Missing data:** compare median imputation with `IterativeImputer` or with adding "was missing" indicator features.

## Data and credits

Pima Indians Diabetes Database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases
(Smith et al., 1988), distributed via the UCI Machine Learning Repository and Kaggle.
