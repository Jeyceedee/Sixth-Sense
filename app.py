from flask import Flask, render_template, url_for, request, redirect, flash
from braintumorclass import *
import urllib.request
import os
import numpy as np
import pandas as pd
import cv2
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'C:/Users/Carlo/Desktop/SoftEngFlask/static/'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
@app.route("/home")
def index():
    return render_template('index.html')


@app.route("/login")
def login():
    return render_template('login.html')


@app.route("/sidebar")
def sidebar():
    return render_template('sidebar.html')


# Content Pages

@app.route("/admindashboard")
def adminDashboard():
    return render_template('adminDashboard.html')


@app.route("/patientRegistration")
def patientRegistration():
    return render_template('patientRegistration.html')

@app.route("/adminPatient")
def adminPatient():
    return render_template('adminPatient.html')

@app.route("/patient_statusEdit")
def patient_statusEdit():
    return render_template('patient_statusEdit.html')

@app.route("/patient_statusView")
def patient_statusView():
    return render_template('patient_statusView.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(img):
    model.load_weights('./Xception_best_weights2.h5')
    p = model.predict(img)
    p = np.argmax(p,axis=1)[0]

    if p==0:
        p='Glioma Tumor'
    elif p==1:
        p='The model predicts that there is no tumor'
    elif p==2:
        p='Meningioma Tumor'
    else:
        p='Pituitary Tumor'
    if p!=1:
        return p


@app.route('/classification', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            img = cv2.imread(filename)
            opencvImage = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(opencvImage,(150,150))
            img = img.reshape(1,150,150,3)
            p = predict(img)
            return render_template('result.html', img=file.filename, p=p)
    else:
        return render_template('classification.html')


@app.route("/adminSettings")
def adminSettings():
    return render_template('adminSettings.html')

@app.route("/doctorList")
def doctorList():
    return render_template('doctorsList.html')




if __name__ == "__main__":
    app.run(debug=True)
