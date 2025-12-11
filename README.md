🩺 AI Disease Prediction System
💡 Overview

A fast, interactive, multi-disease prediction platform built with a full-stack Python architecture.
The system combines a modern Streamlit frontend with a high-performance FastAPI backend, supporting real-time ML predictions and powerful data analysis tools.

🚀 Key Features
1. 📊 Multi-Disease Prediction (Real-Time)

Instant and accurate ML-powered predictions for:

Diabetes

Lung Cancer

Liver Disease

Stroke

Highlights:

Optimized scikit-learn models (Random Forest, Logistic Regression, etc.)

Asynchronous FastAPI routes for high speed

Models loaded once in memory for ultra-fast inference

2. 🖥️ Modern Multi-Page Streamlit UI

A clean, intuitive interface with dedicated pages:

Home (overview)

Disease Prediction Pages

Data Analysis Dashboard

About

Design:

Custom light theme

Organized input layout

Smooth navigation across pages

3. ✨ Intelligent & Validated Forms

Each form is tailored to disease-specific features.

Real-time validation via Pydantic

Clear success/error messages

Clean mapping between user input and ML models

4. 🔥 Optimized FastAPI Backend

Designed for performance and scalability.

Async endpoints:
/predict/diabetes, /predict/liver, /predict/stroke, /predict/cancer

Persistent model loading via joblib

Modular architecture for adding new models easily

5. 📈 Built-In Data Analysis Dashboard

Upload CSV files and explore data instantly.

Summary statistics

Missing value reports

Clean Pandas tables

Visualizations (Correlation heatmaps, histograms, bar/line charts)