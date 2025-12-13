import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

# =====================================================
# Load & Clean Dataset
# =====================================================
df = pd.read_csv("liver_dataset.csv", encoding="latin1")
df.columns = df.columns.str.strip()

df = df.rename(columns={
    'Age of the patient': 'Age_of_the_patient',
    'Gender of the patient': 'Gender_of_the_patient',
    'Total Bilirubin': 'Total_Bilirubin',
    'Direct Bilirubin': 'Direct_Bilirubin',
    'Alkphos Alkaline Phosphotase': 'Alkphos_Alkaline_Phosphotase',
    'Sgpt Alamine Aminotransferase': 'Sgpt_Alamine_Aminotransferase',
    'Sgot Aspartate Aminotransferase': 'Sgot_Aspartate_Aminotransferase',
    'Total Protiens': 'Total_Protiens',
    'ALB Albumin': 'ALB_Albumin',
    'A/G Ratio Albumin and Globulin Ratio': 'AG_Ratio_Albumin_and_Globulin_Ratio',
    'Result': 'Result'
})

# =====================================================
# Encode Gender
# =====================================================
gender_encoder = LabelEncoder()
df['Gender_of_the_patient'] = gender_encoder.fit_transform(df['Gender_of_the_patient'])

# =====================================================
# Features & Target
# =====================================================
X = df[
    [
        'Age_of_the_patient',
        'Gender_of_the_patient',
        'Total_Bilirubin',
        'Direct_Bilirubin',
        'Alkphos_Alkaline_Phosphotase',
        'Sgpt_Alamine_Aminotransferase',
        'Sgot_Aspartate_Aminotransferase',
        'Total_Protiens',
        'ALB_Albumin',
        'AG_Ratio_Albumin_and_Globulin_Ratio'
    ]
]

# 🔴 Important: Convert labels from {1,2} → {0,1} for XGBoost
y = df['Result'].map({1: 0, 2: 1})
print("Target classes:", y.unique())

# =====================================================
# Train / Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# Models & Hyperparameter Grids
# =====================================================
models = {
    "RandomForest": (
        RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
    ),

    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    ),

    "XGBoost": (
        XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5],
            'classifier__subsample': [0.8, 1.0]
        }
    )
}

# =====================================================
# Training Loop
# =====================================================
best_model = None
best_score = 0

for name, (model, params) in models.items():
    print(f"\n==============================")
    print(f"Training {name}")
    print(f"==============================")

    pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', model)
    ])

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=20,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = search.best_estimator_

# =====================================================
# Save Best Model
# =====================================================
os.makedirs("backend/models", exist_ok=True)

joblib.dump(
    {
        "model": best_model,
        "gender_encoder": gender_encoder
    },
    "backend/models/liver_model.pkl"
)

print(f"\n✅ BEST MODEL SAVED")
print(f"Final Accuracy: {best_score:.4f}")
print("File: backend/models/liver_model.pkl")
