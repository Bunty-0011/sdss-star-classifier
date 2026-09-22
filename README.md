<div align="center">

# 🌌 SDSS Stellar Classifier

**Classifying deep-sky objects as GALAXY, QSO, or STAR — powered by LightGBM**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://sdss-star-classifier-dajrmsxy2wdfpevjse6hoh.streamlit.app/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-9ACD32)](https://lightgbm.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-96.8%25-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

**[🚀 Try the Live App](https://sdss-star-classifier-dajrmsxy2wdfpevjse6hoh.streamlit.app/)**

</div>

---

## 📖 Overview

This project classifies sky objects observed by the **Sloan Digital Sky Survey (SDSS)** into one of three categories — **GALAXY**, **QSO** (quasar), or **STAR** — using photometric and spectral measurements. The final model is a tuned **LightGBM** classifier, wrapped in an interactive **Streamlit** app for real-time predictions.

| Metric | Score |
|---|---|
| **Test Accuracy** | **0.968** |
| Macro F1 | ≈ 0.96 |
| Weighted F1 | ≈ 0.97 |
| GALAXY recall | 0.98 |
| QSO recall | 0.96 |
| STAR recall | 0.92 |

**Dataset:** ~577K rows · GALAXY (65.4%) · QSO (20.3%) · STAR (14.3%)

---

## ✨ Features

- 🔭 Predicts object class from raw photometric bands (`u, g, r, i, z`), sky position (`alpha, delta`), `redshift`, and `spectral_type`
- 📊 Shows per-class prediction probabilities with an interactive bar chart
- ⚙️ Full training pipeline included and reproducible end-to-end
- 🧪 Rigorously validated — leakage-checked, class-imbalance-aware, cross-validated

---

## 🗂️ Repository Structure

```
sdss-star-classifier/
├── app.py                     # Streamlit app — loads the model & serves predictions
├── train_and_save_model.py    # Full training pipeline (data → features → tuning → model.pkl)
├── model.pkl                  # Trained LightGBM pipeline (preprocessing + model)
├── label_encoder.pkl          # Encodes/decodes GALAXY / QSO / STAR labels
├── requirements.txt           # Python dependencies
├── star_classification.ipynb  # Full notebook: EDA, feature engineering, leakage checks, evaluation
└── .gitignore
```

---

## 🧠 Model Development Notes

Started at **~0.9618** accuracy with a baseline pipeline, pushed to **0.968** through:

- 🔍 **Feature pruning** based on LightGBM gain-importance — dropped weak raw bands/products
- 🎯 **StratifiedKFold** cross-validation + a wider `RandomizedSearchCV` grid
- ❌ **Ruled out two things that didn't help:** Optuna (didn't beat RandomizedSearchCV on this search budget), and clipping small negative `redshift` values (turned out to be measurement noise carrying real signal — clipping slightly *hurt* accuracy)
- ✅ **Checked `galaxy_population` for target leakage** via crosstab against `class` — confirmed genuine feature, not a leak
- 🔬 Remaining errors concentrate in **STAR↔GALAXY confusion**, tied to a subset of STAR objects with anomalously galaxy-like `redshift`

📓 See [`star_classification.ipynb`](./star_classification.ipynb) for the full walkthrough — EDA, correlation heatmap, confusion matrix, and feature importance plots.

---

## 🚀 Getting Started

### Run the app locally

```bash
# 1. Clone the repo
git clone https://github.com/Bunty-0011/sdss-star-classifier.git
cd sdss-star-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Retrain the model from scratch

1. Download the SDSS stellar classification dataset (see [Dataset](#-dataset) below)
2. Place it as `kaggle1.csv` in this folder
3. Run:
   ```bash
   python train_and_save_model.py
   ```
   This regenerates `model.pkl` and `label_encoder.pkl`.

---

## 📊 Dataset

This project uses a Kaggle **SDSS Stellar Classification** dataset. `kaggle1.csv` is **not included** in this repo (kept lightweight) — download it from Kaggle and place it here if you want to retrain the model.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Model** | LightGBM |
| **Pipeline / Tuning** | scikit-learn (`Pipeline`, `ColumnTransformer`, `RandomizedSearchCV`, `StratifiedKFold`) |
| **App** | Streamlit |
| **EDA** | pandas, seaborn, matplotlib, ydata-profiling |

---

<div align="center">

Made with 🔭 and a lot of hyperparameter tuning

</div>
