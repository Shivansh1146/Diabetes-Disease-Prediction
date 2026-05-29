# Diabetes Disease Prediction System

A complete production-ready Python Full Stack application for early prediction of Diabetes using Machine Learning.

## Features
- **Authentication:** Register, Login, Secure Password Hashing.
- **Machine Learning:** Random Forest vs Logistic Regression comparison. Auto-selects the best model.
- **Modern UI:** Glassmorphism, Dark/Light Mode, Fully Responsive.
- **Reports:** Download PDF reports and Export CSV histories.
- **Admin Panel:** Manage users and view global predictions.
- **Dashboard:** Interactive charts using Chart.js.

## Installation Guide

1. **Navigate to the project folder:**
   ```bash
   cd "c:\Users\shivansh\Desktop\Diabetes Disease Prediction\DiabetesPrediction"
   ```

2. **(Optional but recommended) Create a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Run Instructions

1. **Train the Machine Learning Model:**
   This will generate the synthetic Pima Indians dataset, train the models, select the best one, and save it in the `models/` directory.
   ```bash
   python train_model.py
   ```

2. **Run the Flask Application:**
   This will start the web server on `http://127.0.0.1:5000/`. The database schema will be automatically created on the first run.
   ```bash
   python app.py
   ```

3. **Access the application:**
   Open your browser and navigate to `http://127.0.0.1:5000/`
   
   *Default Admin Credentials:*
   - **Username:** admin
   - **Password:** admin123
