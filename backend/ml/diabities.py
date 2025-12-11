import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("diab.csv")
print(df.columns)

# Features and target
X = df[['Pregnancies','Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age','Race']]
y = df['Outcome']

# Identify categorical columns
categorical_features = ['Race']

# Preprocessing: encode categorical only
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'  # keep numeric columns as is
)

# Create pipeline with preprocessing and RandomForest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter grid for RandomizedSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
search = RandomizedSearchCV(pipeline, param_grid, n_iter=20, cv=5, verbose=2, n_jobs=-1, random_state=42)
search.fit(X_train, y_train)

# Best parameters and model evaluation
print("Best parameters:", search.best_params_)
y_pred = search.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(search.best_estimator_, "backend/models/diabetes_model.pkl")
print("Diabetes model saved inside 'models' folder!")