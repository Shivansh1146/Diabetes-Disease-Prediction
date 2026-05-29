"""utils/helpers.py – shared helper utilities for the Diabetes Prediction System."""

import math


# ─────────────────────────────────────────────────────────────
# Risk / Result Interpretation
# ─────────────────────────────────────────────────────────────

def get_risk_level(probability: float) -> str:
    """Return a human-readable risk level string."""
    pct = probability * 100
    if pct < 25:
        return "Very Low Risk"
    elif pct < 45:
        return "Low Risk"
    elif pct < 60:
        return "Moderate Risk"
    elif pct < 75:
        return "High Risk"
    else:
        return "Very High Risk"


def get_risk_color(probability: float) -> str:
    """Return a Bootstrap / CSS colour class for the risk level."""
    pct = probability * 100
    if pct < 25:
        return "success"
    elif pct < 45:
        return "info"
    elif pct < 60:
        return "warning"
    else:
        return "danger"


def get_result_interpretation(prediction: int, probability: float, data: dict) -> str:
    """Return a plain-English interpretation of the prediction result."""
    pct = round(probability * 100, 1)
    glucose  = data.get('glucose', 0)
    bmi      = data.get('bmi', 0)
    age      = data.get('age', 0)

    factors = []
    if glucose > 140:
        factors.append("elevated glucose levels")
    if bmi > 30:
        factors.append("high BMI")
    if age > 45:
        factors.append("age above 45")

    factor_text = ", ".join(factors) if factors else "the provided clinical data"

    if prediction == 1:
        return (
            f"Based on the analysis, the model indicates a {pct}% probability of diabetes. "
            f"Key contributing factors include {factor_text}. "
            "This result suggests that you may be at risk and should consult a healthcare provider immediately."
        )
    else:
        return (
            f"Based on the analysis, the model indicates a {pct}% probability of diabetes – "
            f"which is relatively low. However, {factor_text} should be monitored regularly. "
            "Continue maintaining a healthy lifestyle and schedule routine check-ups."
        )


def get_health_recommendations(prediction: int, data: dict) -> list:
    """Return a list of personalised health recommendations."""
    glucose  = data.get('glucose', 0)
    bmi      = data.get('bmi', 0)
    age      = data.get('age', 0)
    bp       = data.get('blood_pressure', 0)

    recs = []

    # Common recommendations
    recs.append("💧 Stay hydrated – drink at least 8 glasses of water daily.")
    recs.append("🚶 Exercise regularly – aim for at least 30 minutes of moderate activity 5 days a week.")
    recs.append("🥗 Eat a balanced diet rich in vegetables, whole grains, and lean protein.")
    recs.append("😴 Ensure 7–9 hours of quality sleep every night.")

    # Conditional recommendations
    if glucose > 140:
        recs.append("🍬 Limit sugar and refined carbohydrate intake; monitor blood glucose regularly.")
    if bmi > 30:
        recs.append("⚖️  Work towards a healthy weight – even a 5–10 % reduction can significantly lower risk.")
    if bmi > 25:
        recs.append("🥦 Consider a low-glycaemic-index diet to manage blood sugar spikes.")
    if bp > 90:
        recs.append("❤️  Monitor blood pressure; reduce sodium intake and manage stress.")
    if age > 45:
        recs.append("🩺 Schedule annual diabetes screening as part of your preventive health care.")
    if prediction == 1:
        recs.append("🏥 Consult an endocrinologist or diabetes specialist as soon as possible.")
        recs.append("💊 Discuss possible medication or insulin therapy with your doctor.")
        recs.append("📊 Keep a daily log of blood glucose, diet, and exercise.")

    return recs


# ─────────────────────────────────────────────────────────────
# BMI Calculator
# ─────────────────────────────────────────────────────────────

def calculate_bmi(weight_kg: float, height_cm: float) -> dict:
    """Calculate BMI and return value + category."""
    if height_cm <= 0 or weight_kg <= 0:
        return {"bmi": 0, "category": "Invalid input", "color": "secondary"}

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 18.5:
        category, color = "Underweight", "info"
    elif bmi < 25:
        category, color = "Normal weight", "success"
    elif bmi < 30:
        category, color = "Overweight", "warning"
    elif bmi < 35:
        category, color = "Obese (Class I)", "danger"
    elif bmi < 40:
        category, color = "Obese (Class II)", "danger"
    else:
        category, color = "Obese (Class III)", "danger"

    return {"bmi": bmi, "category": category, "color": color}


# ─────────────────────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────────────────────

FIELD_RANGES = {
    "pregnancies":               (0,   20,   "Pregnancies"),
    "glucose":                   (50,  250,  "Glucose"),
    "blood_pressure":            (30,  140,  "Blood Pressure"),
    "skin_thickness":            (0,   100,  "Skin Thickness"),
    "insulin":                   (0,   900,  "Insulin"),
    "bmi":                       (10,  70,   "BMI"),
    "diabetes_pedigree_function":(0.0, 2.5,  "Diabetes Pedigree Function"),
    "age":                       (1,   120,  "Age"),
}


def validate_prediction_input(form_data: dict) -> list:
    """Validate form input; return list of error strings (empty = valid)."""
    errors = []
    for field, (lo, hi, label) in FIELD_RANGES.items():
        raw = form_data.get(field, "")
        if raw == "" or raw is None:
            errors.append(f"{label} is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not (lo <= val <= hi):
            errors.append(f"{label} must be between {lo} and {hi}.")
    return errors


def calibrate_clinical_probability(raw_prob: float, data: dict, prediction: int) -> float:
    """
    Calibrates statistical Random Forest probability using standard clinical guidelines.
    Corrects for tree-split axis-aligned flattening and survival biases (e.g., extreme age with high glucose).
    """
    glucose = data.get('glucose', 0)
    bmi = data.get('bmi', 0)
    
    calibrated_prob = raw_prob
    
    # 1. Mild logit expansion to spread Random Forest vote frequencies away from the noisy 0.5 cluster
    if 0.01 < calibrated_prob < 0.99:
        logit = math.log(calibrated_prob / (1.0 - calibrated_prob))
        calibrated_prob = 1.0 / (1.0 + math.exp(-logit * 1.35))
        
    # 2. Clinical Guideline Overlay (WHO / ADA Standards)
    # Random Blood Glucose >= 200 mg/dL is diagnostic of diabetes by itself.
    if glucose >= 200:
        if prediction == 1:
            calibrated_prob = max(calibrated_prob, 0.95 + (glucose - 200) * 0.0008)
            calibrated_prob = min(calibrated_prob, 0.999)
            
    # Impaired Glucose Tolerance (140 - 199 mg/dL)
    elif glucose >= 140:
        if prediction == 1:
            boost = 0.12
            if bmi >= 35:
                boost += 0.06
            calibrated_prob = min(calibrated_prob + boost, 0.93)
            
    # Highly obese with elevated glucose
    if bmi >= 45 and glucose >= 120 and prediction == 1:
        calibrated_prob = max(calibrated_prob, 0.88)
        
    # Ensure strict consistency between label and calibrated probability
    if prediction == 1:
        calibrated_prob = max(calibrated_prob, 0.501)
    else:
        calibrated_prob = min(calibrated_prob, 0.499)
        
    return calibrated_prob
