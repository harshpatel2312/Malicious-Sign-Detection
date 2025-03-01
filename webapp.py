from flask import Flask, render_template
import requests
import plotly.graph_objects as go
import plotly.io as pio

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
    macro_precision = metrics.get("Macro_Precision", 0)*100
    macro_recall = metrics.get("Macro_Recall", 0)*100
    macro_f1_score = metrics.get("Macro_F1_score", 0)*100
    red_support = metrics.get("Red_Support", 0)
    yellow_support = metrics.get("Yellow_Support", 0)
    green_support = metrics.get("Green_Support", 0)
    unknown_support = metrics.get("Unknown_Support", 0)
    return render_template("monitor.html", metrics=metrics)

if __name__=="__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)