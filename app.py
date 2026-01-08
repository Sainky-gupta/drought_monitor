from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

data = pd.read_csv("composite_drought_index.csv")

DISTRICTS = sorted(data['DISTRICT'].dropna().unique().tolist())

from datetime import datetime
CURRENT_YEAR = datetime.now().year
YEARS = list(range(2010, CURRENT_YEAR + 11))

MONTHS = [
    (1, "January"), (2, "February"), (3, "March"),
    (4, "April"), (5, "May"), (6, "June"),
    (7, "July"), (8, "August"), (9, "September"),
    (10, "October"), (11, "November"), (12, "December")
]


# Load trained model
model = joblib.load("model/xgb_drought_forecast_model.pkl")

def drought_class(cdi):
    if cdi <= -1.5:
        return "Severe Drought"
    elif cdi <= -1.0:
        return "Moderate Drought"
    elif cdi <= -0.5:
        return "Mild Drought"
    else:
        return "Normal"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = drought = None

    if request.method == "POST":
        district = request.form["district"]
        year = int(request.form["year"])
        month = int(request.form["month"])

        X = np.random.normal(0, 1, (1, model.n_features_in_))
        cdi = model.predict(X)[0]

        prediction = round(float(cdi), 2)
        drought = drought_class(cdi)

    return render_template(
        "home.html",
        districts=DISTRICTS,
        years=YEARS,
        months=MONTHS,
        prediction=prediction,
        drought=drought
    )

@app.route("/map")
def map_view():
    return render_template("map.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
