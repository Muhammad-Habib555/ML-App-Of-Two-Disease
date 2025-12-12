🩺 AI Disease Prediction System

💡 Overview
A fast, interactive AI-powered platform for predicting Diabetes and Liver Disease. Built with Streamlit frontend and FastAPI backend, it supports real-time ML predictions and provides a data analysis dashboard for exploring datasets.

🚀 Key Features
1. 📊 Real-Time Disease Prediction

Predict Diabetes and Liver Disease instantly with ML models.
Highlights:

Optimized scikit-learn models (Random Forest, Logistic Regression, etc.)

Asynchronous FastAPI endpoints for ultra-fast inference

Models loaded once in memory for high performance

2. 🖥️ Modern Multi-Page Streamlit UI

Clean, intuitive interface with dedicated pages:

Home – Overview & introduction

Disease Prediction – Diabetes & Liver

Data Analysis Dashboard – Explore CSV datasets

About – Project info

Design Features:

Custom light theme 🎨

Organized input layout 📝

Smooth multi-page navigation 🔄

3. ✨ Intelligent & Validated Forms

Forms tailored to specific disease features

Real-time validation via Pydantic ✅

Clear success/error messages 💬

Seamless mapping between user input and ML models 🔗

4. 🔥 Optimized FastAPI Backend

Async endpoints:

/predict/diabetes

/predict/liver

Persistent model loading with joblib 💾

Modular architecture for easy model addition 🛠️

5. 📈 Built-In Data Analysis Dashboard

Upload CSV files and explore datasets instantly

Summary statistics & missing value reports

Clean Pandas tables 📋

Visualizations: heatmaps, histograms, bar/line charts 📊

⚡ Tech Stack

Frontend: Streamlit 🌐

Backend: FastAPI ⚡

ML: scikit-learn 🤖

Data Processing: Pandas, NumPy 🐼
