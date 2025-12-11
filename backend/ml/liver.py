import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("liver_dataset.csv", encoding='latin1')

# Strip leading/trailing whitespace from column names
df.columns = df.columns.str.strip()

# Rename columns to match backend/frontend snake_case
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

# Encode categorical column
le = LabelEncoder()
df['Gender_of_the_patient'] = le.fit_transform(df['Gender_of_the_patient'])

# Features and target
X = df[['Age_of_the_patient', 'Gender_of_the_patient', 'Total_Bilirubin', 
        'Direct_Bilirubin', 'Alkphos_Alkaline_Phosphotase', 'Sgpt_Alamine_Aminotransferase', 
        'Sgot_Aspartate_Aminotransferase', 'Total_Protiens', 'ALB_Albumin', 
        'AG_Ratio_Albumin_and_Globulin_Ratio']]

y = df['Result']

# Check dataset
print("Shape of X:", X.shape)
print("Columns in X:", X.columns.tolist())
print("First 5 rows:\n", X.head())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model
os.makedirs("backend/models", exist_ok=True)
joblib.dump(model, "backend/models/liver_model.pkl")
print("Liver model saved!")
