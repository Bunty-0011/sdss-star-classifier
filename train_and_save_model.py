import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import lightgbm as lgb

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv('kaggle1.csv')

# ---------------------------------------------------------------
# 2. Encode target
# ---------------------------------------------------------------
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# ---------------------------------------------------------------
# 3. Feature engineering (must exactly match what app.py does at inference time)
# ---------------------------------------------------------------
df['u_g'] = df['u'] - df['g']
df['g_r'] = df['g'] - df['r']
df['r_i'] = df['r'] - df['i']
df['i_z'] = df['i'] - df['z']
df['ug'] = df['u'] * df['g']
df['gr'] = df['g'] * df['r']
df['ri'] = df['r'] * df['i']
df['iz'] = df['i'] * df['z']

X = df.drop(columns=['class'])
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 4. Preprocessing (pruned feature set — see notebook for why)
# ---------------------------------------------------------------
numerical_features_v2 = [
    'alpha', 'delta', 'u', 'g', 'z', 'redshift',
    'u_g', 'g_r', 'r_i', 'i_z', 'ug', 'iz'
]
categorical_features = ['spectral_type']

preprocessor_v2 = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features_v2),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ---------------------------------------------------------------
# 5. Pipeline + hyperparameter search (same as notebook)
# ---------------------------------------------------------------
clf = Pipeline(steps=[
    ("preprocessor", preprocessor_v2),
    ("model", lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        metric="multi_logloss",
        boosting_type="gbdt",
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    "model__n_estimators": [100, 200, 300, 400, 500, 600, 800, 1000],
    "model__learning_rate": [0.005, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    "model__max_depth": [-1, 3, 4, 5, 6, 8, 10, 12],
    "model__num_leaves": [15, 20, 31, 40, 63, 80, 100, 127, 150],
    "model__min_child_samples": [5, 10, 20, 30, 50, 70, 100],
    "model__subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
    "model__reg_alpha": [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10],
    "model__reg_lambda": [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10],
    "model__min_split_gain": [0, 0.01, 0.05, 0.1, 0.2],
}

random_search = RandomizedSearchCV(
    clf,
    param_distributions=param_dist,
    n_iter=50,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    random_state=42,
    verbose=2
)

print("Training... this will take a while.")
random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

from sklearn.metrics import accuracy_score
y_pred = random_search.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------------------------------------------
# 6. Save the trained pipeline + label encoder to disk
# ---------------------------------------------------------------
# IMPORTANT: we save `best_estimator_` — the full Pipeline (preprocessor + model)
# so app.py doesn't need to redo any preprocessing logic manually.
best_model = random_search.best_estimator_

joblib.dump(best_model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("\nSaved: model.pkl, label_encoder.pkl")
print("These two files, along with app.py and requirements.txt, are what you deploy.")