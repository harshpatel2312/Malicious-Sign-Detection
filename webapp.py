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

    fig1 = go.Figure(data=[go.Pie(
        labels=["Green", "Yellow", "Red", "Unknown"],
        values=[green_support, yellow_support, red_support, unknown_support],
        hole=0.4
    )])
    fig1.update_layout(title_text="Macro Support")

    fig2 = go.Figure(data=[go.Pie(
        labels=["Macro Precision", "Remaining"],
        values=[macro_precision, 100 - macro_precision],
        hole=0.4
    )])
    fig2.update_layout(title_text="Macro Precision")

    fig3 = go.Figure(data=[go.Pie(
        labels=["Macro Recall", "Remaining"],
        values=[macro_recall, 100 - macro_recall],
        hole=0.4
    )])
    fig3.update_layout(title_text="Macro Recall")

    chart1_html = pio.to_html(fig1, full_html=False)
    chart2_html = pio.to_html(fig2, full_html=False)
    chart3_html = pio.to_html(fig3, full_html=False)

    return render_template("monitor.html",
                           metrics=metrics,
                           chart1_html=chart1_html,
                           chart2_html=chart2_html,
                           chart3_html=chart3_html)

if __name__=="__main__":
    app.run(host='127.0.0.1', port=5001, debug=True)