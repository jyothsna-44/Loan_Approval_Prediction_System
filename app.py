from flask import Flask, render_template, request, redirect, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np

app = Flask(__name__,
            template_folder='loan project/templates',
            static_folder='loan project/static')

app.secret_key = "major_project_super_key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///loan_system.db'
db = SQLAlchemy(app)

# ---------------- LOAD MODEL ----------------
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------- DATABASE ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))


class LoanApp(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    amount = db.Column(db.Float)
    status = db.Column(db.String(20))
    risk = db.Column(db.String(20))


with app.app_context():
    db.create_all()

# ---------------- ROUTES ----------------

@app.route('/')
def login_page():
    return render_template('login.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


# ---------------- LOGIN ----------------
@app.route('/handle_login', methods=['POST'])
def handle_login():
    user = User.query.filter_by(email=request.form['email']).first()

    if user and check_password_hash(user.password, request.form['password']):
        session['u_id'] = user.id
        session['u_name'] = user.name
        return redirect(url_for('dashboard'))

    return render_template('login.html', error="Invalid Credentials")


# ---------------- REGISTER ----------------
@app.route('/handle_register', methods=['POST'])
def handle_register():
    try:
        hashed_password = generate_password_hash(request.form['password'])

        new_user = User(
            name=request.form['name'],
            email=request.form['email'],
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        return redirect('/')

    except:
        return render_template('register.html', error="User already exists")


# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
def dashboard():
    if 'u_id' not in session:
        return redirect('/')

    history = LoanApp.query.filter_by(user_id=session['u_id']).all()
    approved = LoanApp.query.filter_by(user_id=session['u_id'], status="Approved").count()
    rejected = LoanApp.query.filter_by(user_id=session['u_id'], status="Rejected").count()

    return render_template('dashboard.html',
                           apprv=approved,
                           rejct=rejected,
                           history=history)


# ---------------- PREDICT PAGE ----------------
@app.route('/predict')
def predict_page():
    if 'u_id' not in session:
        return redirect('/')
    return render_template('predict.html')


# ---------------- ANALYZE ----------------
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'u_id' not in session:
        return redirect('/')

    try:
        # Collect ALL inputs (must match training order)
        data = [
            float(request.form['Gender']),
            float(request.form['Married']),
            float(request.form['Dependents']),
            float(request.form['Education']),
            float(request.form['Self_Employed']),
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            float(request.form['Credit_History']),
            float(request.form['Property_Area'])
        ]

        # Scale data
        data_scaled = scaler.transform([data])

        # Prediction probability
        prob = model.predict_proba(data_scaled)[0][1]

        # Result
        result = "Approved" if prob >= 0.5 else "Rejected"

        # Risk level
        if prob >= 0.8:
            risk = "Low Risk"
        elif prob >= 0.5:
            risk = "Medium Risk"
        else:
            risk = "High Risk"

        # Save to database
        log = LoanApp(
            user_id=session['u_id'],
            amount=float(request.form['LoanAmount']),
            status=result,
            risk=risk
        )

        db.session.add(log)
        db.session.commit()

        return render_template('predict.html',
                               result=result,
                               risk=risk,
                               score=round(prob * 100, 2))

    except Exception as e:
        return render_template('predict.html',
                               result=f"Error: {str(e)}")


# ---------------- LOGOUT ----------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')


# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)