"""
app.py – Diabetes Disease Prediction System
Flask backend with SQLite, Flask-Login, ML inference, CSV/PDF export.
"""

import os, io, csv, json, datetime
from functools import wraps

import joblib
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, jsonify, send_file, Response, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

from utils.helpers import (
    get_risk_level, get_risk_color, get_result_interpretation,
    get_health_recommendations, calculate_bmi, validate_prediction_input,
    calibrate_clinical_probability
)
from utils.pdf_generator import generate_patient_pdf

# ═══════════════════════════════════════════════════
# App Initialisation
# ═══════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'diabetes-secret-key-2024-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'database.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'warning'

# ═══════════════════════════════════════════════════
# Load ML Model
# ═══════════════════════════════════════════════════
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'diabetes_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

model, scaler, metrics = None, None, {}

def load_model():
    global model, scaler, metrics
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("[OK]  ML model loaded.")
    else:
        print("[!]   No trained model found. Run: python train_model.py")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

# Helper to get IST time (naive datetime for SQLite storage)
def get_ist_time():
    IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    return datetime.datetime.now(IST).replace(tzinfo=None)


# ═══════════════════════════════════════════════════
# Database Models
# ═══════════════════════════════════════════════════
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin      = db.Column(db.Boolean, default=False)
    created_at    = db.Column(db.DateTime, default=get_ist_time)
    predictions   = db.relationship('Prediction', backref='user', lazy=True,
                                    cascade='all, delete-orphan')

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class Prediction(db.Model):
    __tablename__             = 'predictions'
    id                        = db.Column(db.Integer, primary_key=True)
    user_id                   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pregnancies               = db.Column(db.Float)
    glucose                   = db.Column(db.Float)
    blood_pressure            = db.Column(db.Float)
    skin_thickness            = db.Column(db.Float)
    insulin                   = db.Column(db.Float)
    bmi                       = db.Column(db.Float)
    diabetes_pedigree_function= db.Column(db.Float)
    age                       = db.Column(db.Integer)
    result                    = db.Column(db.Integer)  # 0 = No, 1 = Yes
    risk_percentage           = db.Column(db.Float)
    confidence_score          = db.Column(db.Float)
    risk_level                = db.Column(db.String(30))
    timestamp                 = db.Column(db.DateTime, default=get_ist_time)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ═══════════════════════════════════════════════════
# Admin-only decorator
# ═══════════════════════════════════════════════════
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════
# Routes – Auth
# ═══════════════════════════════════════════════════
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Welcome back, {user.username}! 👋', 'success')
            return redirect(next_page or url_for('dashboard'))

        flash('Invalid username or password.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip()
        password  = request.form.get('password', '')
        confirm   = request.form.get('confirm_password', '')

        errors = []
        if len(username) < 3:
            errors.append('Username must be at least 3 characters.')
        if '@' not in email:
            errors.append('Please enter a valid email address.')
        if len(password) < 6:
            errors.append('Password must be at least 6 characters.')
        if password != confirm:
            errors.append('Passwords do not match.')
        if User.query.filter_by(username=username).first():
            errors.append('Username already taken.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')

        if errors:
            for e in errors:
                flash(e, 'danger')
        else:
            user = User(username=username, email=email)
            user.set_password(password)
            # First user becomes admin
            if User.query.count() == 0:
                user.is_admin = True
            db.session.add(user)
            db.session.commit()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ═══════════════════════════════════════════════════
# Routes – Dashboard
# ═══════════════════════════════════════════════════
@app.route('/dashboard')
@login_required
def dashboard():
    total_preds  = Prediction.query.filter_by(user_id=current_user.id).count()
    high_risk    = Prediction.query.filter_by(user_id=current_user.id, result=1).count()
    low_risk     = Prediction.query.filter_by(user_id=current_user.id, result=0).count()
    recent_preds = (Prediction.query
                    .filter_by(user_id=current_user.id)
                    .order_by(Prediction.timestamp.desc())
                    .limit(5).all())

    # Chart: last 6 months monthly counts
    monthly_data = _monthly_chart_data(user_id=current_user.id)

    return render_template(
        'dashboard.html',
        total_preds=total_preds,
        high_risk=high_risk,
        low_risk=low_risk,
        recent_preds=recent_preds,
        monthly_data=monthly_data,
        metrics=metrics,
    )


def _monthly_chart_data(user_id=None):
    """Return last 6 months labels + diabetic/non-diabetic counts."""
    today = get_ist_time()
    labels, diabetic_counts, non_diabetic_counts = [], [], []

    for i in range(5, -1, -1):
        month_start = (today.replace(day=1) - datetime.timedelta(days=i * 30)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        if i == 0:
            month_end = today
        else:
            month_end = (month_start + datetime.timedelta(days=32)).replace(day=1)

        q = Prediction.query.filter(
            Prediction.timestamp >= month_start,
            Prediction.timestamp < month_end,
        )
        if user_id:
            q = q.filter_by(user_id=user_id)

        labels.append(month_start.strftime('%b %Y'))
        diabetic_counts.append(q.filter_by(result=1).count())
        non_diabetic_counts.append(q.filter_by(result=0).count())

    return {
        'labels': labels,
        'diabetic': diabetic_counts,
        'non_diabetic': non_diabetic_counts,
    }


# ═══════════════════════════════════════════════════
# Routes – Prediction
# ═══════════════════════════════════════════════════
@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'GET':
        return render_template('prediction.html')

    if model is None or scaler is None:
        flash('ML model not loaded. Please run: python train_model.py', 'danger')
        return redirect(url_for('predict'))

    # Validate input
    form = request.form
    errors = validate_prediction_input(form)
    if errors:
        for e in errors:
            flash(e, 'danger')
        return render_template('prediction.html', form_data=form)

    # Extract values
    data = {
        'pregnancies':               float(form['pregnancies']),
        'glucose':                   float(form['glucose']),
        'blood_pressure':            float(form['blood_pressure']),
        'skin_thickness':            float(form['skin_thickness']),
        'insulin':                   float(form['insulin']),
        'bmi':                       float(form['bmi']),
        'diabetes_pedigree_function':float(form['diabetes_pedigree_function']),
        'age':                       float(form['age']),
    }

    features = np.array([[
        data['pregnancies'], data['glucose'], data['blood_pressure'],
        data['skin_thickness'], data['insulin'], data['bmi'],
        data['diabetes_pedigree_function'], data['age']
    ]])
    features_scaled = scaler.transform(features)

    prediction   = int(model.predict(features_scaled)[0])
    probabilities= model.predict_proba(features_scaled)[0]
    raw_risk_prob= float(probabilities[1])
    
    # Apply clinical guideline probability calibration (fixes RF noise & survival bias)
    risk_prob    = calibrate_clinical_probability(raw_risk_prob, data, prediction)
    
    confidence   = float(max(risk_prob, 1.0 - risk_prob))
    risk_pct     = round(risk_prob * 100, 1)
    risk_lvl     = get_risk_level(risk_prob)
    risk_clr     = get_risk_color(risk_prob)
    interpretation = get_result_interpretation(prediction, risk_prob, data)
    recommendations = get_health_recommendations(prediction, data)

    # Persist to DB
    pred_record = Prediction(
        user_id=current_user.id,
        pregnancies=data['pregnancies'],
        glucose=data['glucose'],
        blood_pressure=data['blood_pressure'],
        skin_thickness=data['skin_thickness'],
        insulin=data['insulin'],
        bmi=data['bmi'],
        diabetes_pedigree_function=data['diabetes_pedigree_function'],
        age=int(data['age']),
        result=prediction,
        risk_percentage=risk_pct,
        confidence_score=round(confidence * 100, 1),
        risk_level=risk_lvl,
    )
    db.session.add(pred_record)
    db.session.commit()

    return render_template(
        'prediction.html',
        result=prediction,
        risk_percentage=risk_pct,
        confidence_score=round(confidence * 100, 1),
        risk_level=risk_lvl,
        risk_color=risk_clr,
        interpretation=interpretation,
        recommendations=recommendations,
        form_data=form,
        prediction_id=pred_record.id,
    )


# ═══════════════════════════════════════════════════
# Routes – History
# ═══════════════════════════════════════════════════
@app.route('/history')
@login_required
def history():
    page  = request.args.get('page', 1, type=int)
    preds = (Prediction.query
             .filter_by(user_id=current_user.id)
             .order_by(Prediction.timestamp.desc())
             .paginate(page=page, per_page=10, error_out=False))
    return render_template('history.html', preds=preds)


@app.route('/history/delete/<int:pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    db.session.delete(pred)
    db.session.commit()
    flash('Prediction record deleted.', 'info')
    return redirect(request.referrer or url_for('history'))


# ═══════════════════════════════════════════════════
# Routes – Reports / Export
# ═══════════════════════════════════════════════════
@app.route('/export/csv')
@login_required
def export_csv():
    if current_user.is_admin:
        preds = Prediction.query.options(db.joinedload(Prediction.user)).order_by(Prediction.timestamp.desc()).all()
    else:
        preds = (Prediction.query
                 .options(db.joinedload(Prediction.user))
                 .filter_by(user_id=current_user.id)
                 .order_by(Prediction.timestamp.desc()).all())

    si = io.StringIO()
    cw = csv.writer(si)
    
    # Write header
    cw.writerow(['ID','Username','Pregnancies','Glucose','BloodPressure','SkinThickness',
                 'Insulin','BMI','DiabetesPedigreeFunction','Age',
                 'Result','RiskPercentage','ConfidenceScore','RiskLevel','Timestamp'])
                 
    for p in preds:
        cw.writerow([
            p.id, 
            p.user.username if p.user else 'Unknown',
            p.pregnancies, 
            p.glucose, 
            p.blood_pressure,
            p.skin_thickness, 
            p.insulin, 
            p.bmi,
            p.diabetes_pedigree_function, 
            p.age,
            'Diabetic' if p.result == 1 else 'Not Diabetic',
            p.risk_percentage, 
            p.confidence_score,
            p.risk_level, 
            p.timestamp.strftime("%Y-%m-%d %H:%M:%S") if p.timestamp else 'N/A'
        ])
        
    return Response(
        si.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=diabetes_predictions.csv'}
    )


@app.route('/export/pdf/<int:pred_id>')
@login_required
def export_pdf(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    if pred.user_id != current_user.id and not current_user.is_admin:
        abort(403)

    pred_data = {
        'id':                  pred.id,
        'pregnancies':         pred.pregnancies,
        'glucose':             pred.glucose,
        'blood_pressure':      pred.blood_pressure,
        'skin_thickness':      pred.skin_thickness,
        'insulin':             pred.insulin,
        'bmi':                 pred.bmi,
        'diabetes_pedigree':   pred.diabetes_pedigree_function,
        'age':                 pred.age,
        'result':              pred.result,
        'risk_percentage':     pred.risk_percentage,
        'confidence_score':    pred.confidence_score,
        'risk_level':          pred.risk_level,
        'timestamp':           pred.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'interpretation':      get_result_interpretation(pred.result, pred.risk_percentage / 100, {
            'glucose': pred.glucose, 'bmi': pred.bmi,
            'age': pred.age, 'blood_pressure': pred.blood_pressure,
        }),
    }
    pdf_bytes = generate_patient_pdf(pred_data, pred.user.username)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'diabetes_report_{pred_id}.pdf',
    )


# ═══════════════════════════════════════════════════
# Routes – Accuracy / Model Info
# ═══════════════════════════════════════════════════
@app.route('/accuracy')
@login_required
def accuracy():
    return render_template('accuracy.html', metrics=metrics)


# ═══════════════════════════════════════════════════
# Routes – Admin Panel
# ═══════════════════════════════════════════════════
@app.route('/admin')
@login_required
@admin_required
def admin():
    users      = User.query.order_by(User.created_at.desc()).all()
    all_preds  = Prediction.query.order_by(Prediction.timestamp.desc()).limit(50).all()
    total_users= User.query.count()
    total_preds= Prediction.query.count()
    diabetic   = Prediction.query.filter_by(result=1).count()
    monthly    = _monthly_chart_data()
    return render_template(
        'admin.html',
        users=users,
        all_preds=all_preds,
        total_users=total_users,
        total_preds=total_preds,
        diabetic=diabetic,
        monthly_data=monthly,
        metrics=metrics,
    )


@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(user_id):
    if user_id == current_user.id:
        flash("You cannot delete your own account.", 'warning')
        return redirect(url_for('admin'))
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f'User "{user.username}" deleted.', 'success')
    return redirect(url_for('admin'))


@app.route('/admin/delete_prediction/<int:pred_id>', methods=['POST'])
@login_required
@admin_required
def admin_delete_prediction(pred_id):
    pred = Prediction.query.get_or_404(pred_id)
    db.session.delete(pred)
    db.session.commit()
    flash('Prediction deleted.', 'info')
    return redirect(url_for('admin'))


# ═══════════════════════════════════════════════════
# Routes – BMI Calculator (AJAX)
# ═══════════════════════════════════════════════════
@app.route('/api/bmi', methods=['POST'])
@login_required
def api_bmi():
    data = request.get_json()
    result = calculate_bmi(
        float(data.get('weight', 0)),
        float(data.get('height', 0))
    )
    return jsonify(result)


# ═══════════════════════════════════════════════════
# Routes – Info / Static Pages
# ═══════════════════════════════════════════════════
@app.route('/info')
@login_required
def info():
    return render_template('info.html')


@app.route('/about')
@login_required
def about():
    return render_template('about.html')


@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')


# ═══════════════════════════════════════════════════
# Error handlers
# ═══════════════════════════════════════════════════
@app.errorhandler(403)
def forbidden(e):
    return render_template('error.html', code=403,
                           msg='Access Forbidden – You do not have permission.'), 403


@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404,
                           msg='Page not found.'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500,
                           msg='Internal server error.'), 500


# ═══════════════════════════════════════════════════
# Initialization for Gunicorn & Local
# ═══════════════════════════════════════════════════
with app.app_context():
    db.create_all()
    # Create default admin if no users exist
    if User.query.count() == 0:
        admin_user = User(username='admin', email='admin@diabetes.com', is_admin=True)
        admin_user.set_password('admin123')
        db.session.add(admin_user)
        db.session.commit()
        print("[OK]  Default admin created  ->  username: admin  |  password: admin123")

load_model()

# ═══════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
