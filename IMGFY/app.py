from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_login import UserMixin
import os
import cv2
import numpy as np
import uuid
import csv

# Initialize Flask app and extensions
app = Flask(__name__)


# ------------------ Configurations ------------------
app.secret_key = 'your_super_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///imgfy.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

#Models here
# ================================
# User Model
# ================================
class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User  {self.username}>"

# ================================
# Contact Message Model
# ================================
class ContactMessage(db.Model):
    __tablename__ = 'contact_messages'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<ContactMessage from {self.name}>"

# ================================
# Feedback Model
# ================================
class Feedback(db.Model):
    __tablename__ = 'feedbacks'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comments = db.Column(db.Text, nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Feedback by User {self.user_id}>"


# Folder Paths
UF = os.path.join(app.root_path, 'static', 'uploads')
PF = os.path.join(app.root_path, 'static', 'processed')
os.makedirs(UF, exist_ok=True)
os.makedirs(PF, exist_ok=True)

FORGERY_CSV = 'forgery_results.csv'
PROCESSING_CSV = 'processing_results.csv'

 # Define your models here or import them
# with app.app_context():
#     db.create_all()  # Create tables


# Initialize Login Manager
login_manager = LoginManager(app)
# ------------------- User Load ----------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ------------------ Public Pages ------------------ #
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        message = ContactMessage(
            name=request.form['name'],
            email=request.form['email'],
            message=request.form['message']
        )
        db.session.add(message)
        db.session.commit()
        flash('Your message has been sent successfully!', 'success')
        return redirect(url_for('contact'))
    return render_template('contact.html')

# ------------------ Authentication ------------------ #
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        if data['password'] != data['repassword']:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=data['email']).first():
            flash('Email already registered.', 'warning')
            return redirect(url_for('register'))

        new_user = User(
            username=data['username'],
            email=data['email'],
            password=generate_password_hash(data['password'])
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# ------------------ User Dashboard ------------------ #
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

# ------------------ Feedback ------------------ #
@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        fb = Feedback(
            user_id=current_user.id,
            rating=request.form['rating'],
            comments=request.form['comments']
        )
        db.session.add(fb)
        db.session.commit()
        flash('Thanks for your feedback!')
        return redirect(url_for('feedback'))
    return render_template('feedback.html')

# ------------------ Image Processing ------------------ #
@app.route('/image_processing', methods=['GET', 'POST'])
@login_required
def image_processing():
    if request.method == 'POST':
        file = request.files['uploaded_image']
        algorithm = request.form['processing_operation']

        if file and algorithm:
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(file.filename)[1]
            unique_name = f"{uuid.uuid4().hex}{file_ext}"
            original_path = os.path.join(UF, unique_name)
            file.save(original_path)

            image_id = str(uuid.uuid4())
            processed_filename = f"processed_{unique_name}"
            processed_path = os.path.join(PF, processed_filename)

            try:
                from process_image import processing_image
                processed_image = processing_image(original_path, algorithm)

                if processed_image is None or not isinstance(processed_image, np.ndarray):
                    flash("Image processing failed. Algorithm did not return a valid image.")
                    return redirect(url_for('image_processing'))

                success = cv2.imwrite(processed_path, processed_image)
                if not success:
                    print("Failed to write processed image to:", processed_path)

                with open(PROCESSING_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([image_id, unique_name, processed_filename, algorithm])

                return render_template(
                    'processing_results.html',
                    image_id=image_id,
                    original_image=url_for('static', filename=f'uploads/{unique_name}'),
                    processed_image=url_for('static', filename=f'processed/{processed_filename}'),
                    processing_operation=algorithm
                )

            except Exception as e:
                flash(f"Unexpected error during processing: {str(e)}")
                return redirect(url_for('image_processing'))

        else:
            flash("Please upload an image and select an algorithm.")
            return redirect(url_for('image_processing'))

    return render_template('image_processing.html')

# ------------------ Processing Records ------------------ #
@app.route('/processing_records', methods=['GET'])
@login_required
def processing_records():
    data = []
    if os.path.exists(PROCESSING_CSV):
        with open(PROCESSING_CSV, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
    return render_template('processing_records.html', records=data)

# ------------------ Image Forgery ------------------ #
@app.route('/image_forgery', methods=['GET', 'POST'])
@login_required
def image_forgery():
    if request.method == 'POST':
        file = request.files['uploaded_image']
        algorithm = request.form['forgery_detection']

        if file and algorithm:
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(file.filename)[1]
            unique_name = f"{uuid.uuid4().hex}{file_ext}"
            original_path = os.path.join(UF, unique_name)
            file.save(original_path)

            forgery_id = str(uuid.uuid4())
            processed_filename = f"forgery_{unique_name}"
            processed_path = os.path.join(PF, processed_filename)

            try:
                from forgery_image import detect_forgery
                processed_image = detect_forgery(original_path, algorithm)

                if processed_image is None or not isinstance(processed_image, np.ndarray):
                    flash("Forgery detection failed. Algorithm did not return a valid image.")
                    return redirect(url_for('image_forgery'))

                success = cv2.imwrite(processed_path, processed_image)
                if not success:
                    print("Failed to save processed image at:", processed_path)

                with open(FORGERY_CSV, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([forgery_id ,unique_name, processed_filename, algorithm])

                return render_template(
                    'forgery_results.html',
                    forgery_id=forgery_id,
                    uploaded_image=url_for('static', filename=f'uploads/{unique_name}'),
                    processed_image=url_for('static', filename=f'processed/{processed_filename}'),
                    forgery_detection=algorithm
                )

            except Exception as e:
                flash(f"Unexpected error during forgery detection: {str(e)}")
                return redirect(url_for('image_forgery'))

        else:
            flash("Please upload an image and select an algorithm.")
            return redirect(url_for('image_forgery'))

    return render_template('image_forgery.html')

# ------------------ Forgery Records ------------------ #
@app.route('/forgery_records', methods=['GET'])
@login_required
def forgery_records():
    data = []
    if os.path.exists(FORGERY_CSV):
        with open(FORGERY_CSV, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
    return render_template('forgery_records.html', records=data)

# ------------------ Admin Panel ------------------ #
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@gmail.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1090")

@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['email'] == ADMIN_EMAIL and request.form['password'] == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(url_for('admin_dashboard'))
        flash('Invalid admin credentials.', 'danger')
    return render_template('admin_login.html')

@app.route('/admin-dashboard')
def admin_dashboard():
    if not session.get('admin'):
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('admin_login'))

    users = User.query.all()
    contacts = ContactMessage.query.all()
    feedbacks = Feedback.query.join(User).add_columns(
        Feedback.id, Feedback.comments, Feedback.submitted_at,
        User.username.label('name'), User.email
    ).all()

    return render_template('admin_dashboard.html',
                           users=users, contacts=contacts, feedbacks=feedbacks)

@app.route('/admin-logout')
def admin_logout():
    session.pop('admin', None)
    flash('Logged out from admin panel.', 'info')
    return redirect(url_for('admin_login'))

@app.route('/delete-user/<int:id>')
def delete_user(id):
    if session.get('admin'):
        db.session.delete(User.query.get_or_404(id))
        db.session.commit()
        flash('User  deleted.', 'info')
    else:
        flash('Unauthorized access.', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/delete-feedback/<int:id>')
def delete_feedback(id):
    if session.get('admin'):
        db.session.delete(Feedback.query.get_or_404(id))
        db.session.commit()
        flash('Feedback deleted.', 'info')
    else:
        flash('Unauthorized access.', 'danger')
    return redirect(url_for('admin_dashboard'))

@app.route('/delete-contact/<int:id>')
def delete_contact(id):
    if session.get('admin'):
        db.session.delete(ContactMessage.query.get_or_404(id))
        db.session.commit()
        flash('Contact deleted.', 'info')
    else:
        flash('Unauthorized access.', 'danger')
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
