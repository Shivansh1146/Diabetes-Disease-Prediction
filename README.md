# Diabetes Disease Prediction System 🩺

A complete, production-ready Python Full Stack application for the early prediction of Diabetes using Machine Learning.

## ✨ Features
- **Secure Authentication:** User Registration, Login, and Password Hashing (Werkzeug).
- **Machine Learning Pipeline:** Trains both Logistic Regression and Random Forest models on the real Pima Indians Diabetes Dataset, automatically selecting the best one based on AUC-ROC.
- **Advanced Dashboard:** Interactive Chart.js graphs displaying monthly prediction trends and risk distributions.
- **Prediction Engine:** Input clinical data to receive an instant prediction, complete with risk percentage, AI confidence score, and tailored health recommendations.
- **Admin Panel:** Global oversight to manage users, view all platform predictions, and access system-wide analytics.
- **Automated Reports:** Download individual patient prediction reports as PDFs (ReportLab) or export global history to CSV.
- **Modern UI/UX:** Responsive Glassmorphism design with a fully functional **Dark/Light Mode** toggle.

## 🛠️ Technology Stack
- **Backend:** Python, Flask, SQLAlchemy, Flask-Login
- **Machine Learning:** Scikit-learn, Pandas, NumPy, Joblib
- **Frontend:** HTML5, CSS3 (Vanilla), JavaScript, Bootstrap 5, Chart.js
- **Database:** SQLite

## 🚀 Installation & Setup

1. **Clone the repository and navigate into the folder:**
   ```bash
   git clone https://github.com/Shivansh1146/Diabetes-Disease-Prediction.git
   cd Diabetes-Disease-Prediction
   ```

2. **(Optional but recommended) Create a virtual environment:**
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

4. **Train the Machine Learning Model:**
   This script will automatically download the Pima Indians dataset, clean the data, train the ML models, select the highest-performing algorithm, and save it in the `models/` directory.
   ```bash
   python train_model.py
   ```

5. **Start the Flask Server:**
   This will start the local web server and automatically generate the SQLite database.
   ```bash
   python app.py
   ```

6. **Access the application:**
   Open your browser and navigate to: `http://127.0.0.1:5000/`

   *Default Admin Credentials:*
   - **Username:** `admin`
   - **Password:** `admin123`

## 📊 Model Evaluation
You can view the real-time confusion matrix, accuracy, precision, recall, F1 score, and AUC on the `/accuracy` page within the application once logged in.

---
*Built as a comprehensive AI-powered clinical assessment tool.*
