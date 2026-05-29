# Diabetes Disease Prediction System 🩺

A complete, production-ready Python Full Stack application for the early prediction and risk assessment of Diabetes using Machine Learning. Built with clinical safety parameters, dynamic data visualization, and an elegant Glassmorphism responsive user interface.

## 🔗 Live Demo
Access the live deployed application here:  
👉 **[https://diabetes-prediction-ai-5u0i.onrender.com/](https://diabetes-prediction-ai-5u0i.onrender.com/)**

*Default Admin Credentials:*
- **Username:** `admin`
- **Password:** `admin123`

---

## ✨ Features

- **🔐 Secure Authentication:** Seamless user registration and login management utilizing Flask-Login and Werkzeug password hashing (PBKDF2).
- **🧠 Machine Learning Pipeline:** Trains both Logistic Regression and Random Forest models on the real Pima Indians Diabetes Dataset, automatically selecting the highest-performing model based on cross-validated AUC-ROC.
- **📈 Clinical Probability Calibration Overlay:** Standard statistical model probabilities (notoriously noisy in Random Forests) are mapped to clinical curves using WHO and ADA diagnostic benchmarks. Patients with extreme metrics (e.g., Blood Glucose $\ge$ 200 mg/dL) receive accurate high-certainty risk scoring (95%+), overriding standard data constraints and survival biases.
- **📊 Modern Dashboard & Analytics:** Real-time analytics utilizing Chart.js to render monthly diagnostic trends, risk level spreads, and dynamic comparative model metric radar charts on the `/accuracy` panel.
- **🩺 Patient Risk Assessment:** Form interface with comprehensive client-side and server-side range validations. Outputs prediction classes, calibrated risk percentages, confidence metrics, and custom clinical recommendations.
- **👔 Admin Control Center:** Custom `@admin_required` decorators blocking unauthorized users. Admins can view platform analytics, inspect global user prediction histories, and delete records.
- **📄 Professional Exports:** Download individual clinical reports as beautifully styled PDFs (ReportLab) or export global history logs to Microsoft Excel-compatible CSVs (optimized for Gunicorn thread-safety).
- **🎨 Glassmorphic Theme Engine:** Gorgeous custom CSS design with a responsive **Dark/Light Mode** toggle that persists choices across sessions using LocalStorage.
- **🕒 Local Timezone Logging:** Native IST (UTC +5:30) date-time alignment mapped across all database transactions and PDF reports, bypassing traditional UTC cloud-server mismatches.

---

## 🛠️ Technology Stack

- **Backend:** Python, Flask, SQLAlchemy, Flask-Login
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3 (Vanilla custom variables), JavaScript (ES6), Bootstrap 5, Chart.js
- **Database:** SQLite
- **WSGI Production Server:** Gunicorn

---

## 🚀 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shivansh1146/Diabetes-Disease-Prediction.git
   cd Diabetes-Disease-Prediction
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML Pipeline:**
   This downloads the raw Pima Indians dataset, cleans physiologically-impossible zero metrics, clips extreme outliers at the 1st/99th percentiles, and serializes the winning model, scaler, and metrics to `models/`:
   ```bash
   python train_model.py
   ```

5. **Start the Flask Server:**
   This initiates the local development web server and handles database table creation and admin seeding:
   ```bash
   python app.py
   ```

6. **Open your browser:**
   Go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 🌐 Cloud Deployment (Gunicorn & Render)

This application is fully optimized for containerized cloud deployment (e.g., Render, Heroku) with global thread safety:

*   **Procfile:** Configured to run `web: gunicorn app:app`
*   **Database Initialisation:** All SQLite table creation and admin seeding are migrated out of the `__main__` block, ensuring they successfully run under Gunicorn's WSGI context.
*   **Build Command on Render:**
    ```bash
    pip install -r requirements.txt && python train_model.py
    ```
*   **Start Command on Render:**
    ```bash
    gunicorn app:app
    ```

---

## 📊 Model Architecture & Performance

The pipeline currently selects **Random Forest** (200 estimators, max depth 10, balanced class weights) as the default model due to its high generalizability:

- **Accuracy:** `76.62%`
- **AUC-ROC:** `0.8320`
- **Precision:** `65.52%`
- **Recall:** `70.37%`
- **F1-Score:** `67.86%`

*Detailed model comparative stats, confusion matrices, and metrics are fully interactive on the log-in protected `/accuracy` dashboard page.*

---
*Built with passion as a comprehensive AI-powered clinical assessment tool.*
