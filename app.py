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

@app.route("/patientList")
def patientList():
    return render_template('patientList.html')


if __name__ == "__main__":
    app.run(debug=True)
