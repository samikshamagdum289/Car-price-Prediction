from flask import Flask, render_template, request
import pickle as pkl
import numpy as np
import pandas as pd

app = Flask(__name__)
df = pd.read_csv("cleaned_data.xls")
pipe = pkl.load(open("CPP.pkl", "rb"))   #linear regression

#pipe = pkl.load(open("CPP_decision_tree.pkl", "rb"))      #decision tree regression
@app.route("/")
def hello_world():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/car-project")
def car_project():
    companies = sorted(df["company"].unique())
    names = sorted(df["name"].unique())
    fuel_types = sorted(df["fuel_type"].unique())
    return render_template("car-project.html", companies = companies, names = names, fuel_types = fuel_types)

@app.route("/car-project-result")
def car_project_result():
    company = request.args.get("company")
    name = request.args.get("name")
    year = request.args.get("year")
    kms_driven = request.args.get("kms_driven")
    fuel_type = request.args.get("fuel_type")


    myinput = np.array([company, name, year, kms_driven, fuel_type]).reshape(1, 5)
    columns = ["company", "name", "year", "kms_driven", "fuel_type"]
    mydata = pd.DataFrame(columns = columns, data = myinput)
    result = round(pipe.predict(mydata)[0,0], 2)

    return render_template("car-project-result.html", company = company, name = name, year = year, kms_driven = kms_driven, fuel_type = fuel_type, result = result)