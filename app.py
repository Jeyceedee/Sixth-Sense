from flask import Flask, render_template, url_for, request

app = Flask(__name__)


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

@app.route("/adminSettings")
def adminSettings():
    return render_template('adminSettings.html')

@app.route("/doctorList")
def doctorList():
    return render_template('doctorsList.html')




if __name__ == "__main__":
    app.run(debug=True)
