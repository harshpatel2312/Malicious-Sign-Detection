from flask import Flask
from flask import render_template
import requests

app=Flask(__name__)

def get_metrics():
    try:
        response = requests.get('http://127.0.0.1:5000/data')
        return response.json() if response.status_code == 200 else {}
    except requests.exceptions.RequestException as e:
        print(f"Fetching metrics is failed: {e}")
        return {}

@app.route("/")
def render_home():
    return render_template("home.html")

@app.route("/about")
def render_about():
    return render_template("about.html")

@app.route("/monitor")
def render_monitor():
    metrics = get_metrics()
    return render_template("monitor.html", metrics=metrics)

if __name__=="__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)