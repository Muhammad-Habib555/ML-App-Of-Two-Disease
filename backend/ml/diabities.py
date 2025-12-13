import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

from xgboost import XGBClassifier

# =========================
# Custom Outlier Remover
# =========================
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_bounds_ = X.quantile(0.25) - self.factor * (X.quantile(0.75) - X.quantile(0.25))
        self.upper_bounds_ = X.quantile(0.75) + self.factor * (X.quantile(0.75) - X.quantile(0.25))
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

# =========================
# Load Dataset
# =========================
df = pd.read_csv("diab.csv")

X = df[['Pregnancies','Glucose','BloodPressure','BMI',
        'DiabetesPedigreeFunction','Age','Race']]
y = df['Outcome']

categorical_features = ['Race']
numeric_features = X.drop(columns=categorical_features).columns

# =========================
# Preprocessing Pipeline
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('outliers', IQRClipper())
        ]), numeric_features),

        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# =========================
# Models & Hyperparameters
# =========================
models = {
    "RandomForest": (
        RandomForestClassifier(random_state=42),
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

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_model = None
best_score = 0

# =========================
# Training Loop
# =========================
for name, (model, params) in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    search = RandomizedSearchCV(
        pipeline,
        params,
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

    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = search.best_estimator_

# =========================
# Save Best Model
# =========================
joblib.dump(best_model, "backend/models/diabetes_model.pkl")
print(f"\n✅ Best model saved with accuracy: {best_score:.4f}")
