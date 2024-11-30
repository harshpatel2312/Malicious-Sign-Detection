from flask import Flask
from flask import render_template
import requests

app=Flask(__name__)

@app.route("/")
def render_home():
    return render_template("home.html")

@app.route("/monitor")
def render_monitor():
    try:
        response = requests.get('http://127.0.0.1:5000/data')
        metrics = response.json() if response.status_code == 200 else {}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching colors: {e}")
        metrics ={}
    return render_template("monitor.html", metrics=metrics)

@app.route("/about")
def render_about():
    return render_template("about.html")

if __name__=="__main__":
    app.run(debug=True)