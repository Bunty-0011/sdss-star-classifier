# 🌌 SDSS Stellar Classifier

A machine learning web app that classifies sky objects from the Sloan Digital Sky Survey (SDSS) as **GALAXY**, **QSO** (quasar), or **STAR** based on photometric and spectral measurements — built with LightGBM and deployed as an interactive Streamlit app.

**Test accuracy: 0.968** (macro F1 ≈ 0.96, weighted F1 ≈ 0.97)

## Live Demo

🔗 https://sdss-star-classifier-dajrmsxy2wdfpevjse6hoh.streamlit.app/

## Overview

- **Dataset:** SDSS photometric + spectroscopic data (~577K rows), 3 classes: GALAXY (65.4%), QSO (20.3%), STAR (14.3%)
- **Model:** LightGBM multiclass classifier, tuned with `RandomizedSearchCV` + `StratifiedKFold`
- **Features:** Raw photometric bands (`u, g, r, i, z`), position (`alpha, delta`), `redshift`, `spectral_type`, plus engineered color-difference and product features
- **Per-class recall:** GALAXY 0.98, QSO 0.96, STAR 0.92

## What's in this repo

| File | Purpose |
|---|---|
| `app.py` | Streamlit app — loads the trained model and serves predictions |
| `train_and_save_model.py` | Full training pipeline — loads data, engineers features, tunes hyperparameters, saves `model.pkl` + `label_encoder.pkl` |
| `model.pkl` | Trained LightGBM pipeline (preprocessing + model) |
| `label_encoder.pkl` | Encodes/decodes the GALAXY/QSO/STAR labels |
| `requirements.txt` | Python dependencies |
| `*.ipynb` | Full notebook with EDA, feature engineering, leakage checks, and evaluation |

## Model development notes

- Started at ~0.9618 accuracy with a baseline pipeline; reached 0.968 through:
  - Feature pruning based on LightGBM gain-importance (dropped weak raw bands/products)
  - `StratifiedKFold` cross-validation + a wider `RandomizedSearchCV` grid
  - Ruling out a couple of things that *didn't* help: Optuna (didn't beat RandomizedSearchCV here on this budget), and clipping small negative `redshift` values (measurement noise — clipping them slightly hurt accuracy)
- Checked `galaxy_population` for target leakage via crosstab against `class` — confirmed it's a genuine feature, not a leak
- Remaining errors are concentrated in STAR↔GALAXY confusion, linked to a subset of STAR objects with anomalously galaxy-like `redshift` values

See the notebook for the full walkthrough (EDA, correlation heatmap, confusion matrix, feature importance).

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/Bunty-0011/sdss-star-classifier.git
cd sdss-star-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Retraining the model

If you want to retrain from scratch, download the dataset (see below), place it as `kaggle1.csv` in this folder, and run:

```bash
python train_and_save_model.py
```

This regenerates `model.pkl` and `label_encoder.pkl`.

## Dataset

This project uses a Kaggle SDSS stellar classification dataset. `kaggle1.csv` is not included in this repo (kept lightweight) — download the dataset from Kaggle and place it here if you want to retrain the model.

## Tech stack

- **Model:** LightGBM
- **Pipeline / tuning:** scikit-learn (`Pipeline`, `ColumnTransformer`, `RandomizedSearchCV`, `StratifiedKFold`)
- **App:** Streamlit
- **EDA:** pandas, seaborn, ydata-profiling
