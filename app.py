from flask import Flask, render_template
import requests
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

def get_metrics():
    try:
        response = requests.get('http://24.150.183.74:5000/data')
        return response.json() if response.status_code == 200 else {}
    except requests.exceptions.RequestException as e:
        print(f"Fetching metrics failed: {e}")
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

    if metrics:
        accuracy = metrics.get('accuracy', {})
        macro_avg_metrics = metrics.get('macro avg', {})
        green_metrics = metrics.get('green', {})
        red_metrics = metrics.get('red', {})
        yellow_metrics = metrics.get('yellow', {})
        execution_time = metrics.get('Execution_Time_of_Prediction', 0)

        class_distribution_fig = go.Figure(data=[go.Pie(
            labels=["Green", "Yellow", "Red"],
            values=[green_metrics.get('support', 0), yellow_metrics.get('support', 0), red_metrics.get('support', 0)],
            hole=0.4,
            pull=[0.1, 0.1, 0.1],
            marker=dict(colors=["#00FF00", "#FFFF00", "#FF0000"])
        )])

        accuracy_fig = go.Figure(data=[go.Pie(
            labels=["Accuracy", "Error Rate"],
            values=[accuracy * 100, 100 - accuracy * 100],
            hole=0.4,
            pull=[0, 0.1],
            hovertemplate=["<b>%{label}</b><br>Refers to the fraction of correctly classified instances out of all instances<br><extra></extra>", "<b>%{label}</b><br><extra></extra>"]
        )])

        macro_precision_fig = go.Figure(data=[go.Pie(
            labels=["Macro Precision", "Error Rate"],
            values=[macro_avg_metrics.get('precision', 0) * 100, 100 - macro_avg_metrics.get('precision', 0) * 100],
            hole=0.4,
            pull=[0, 0.1],
            hovertemplate=["<b>%{label}</b><br>Refers to fraction of true positive among all the positives<br><extra></extra>", "<b>%{label}</b><br><extra></extra>"]
        )])

        macro_recall_fig = go.Figure(data=[go.Pie(
            labels=["Macro Recall", "Error Rate"],
            values=[macro_avg_metrics.get('recall', 0) * 100, 100 - macro_avg_metrics.get('recall', 0) * 100],
            hole=0.4,
            pull=[0, 0.1],
            hovertemplate=["<b>%{label}</b><br>Refers to fraction of true positive among all correct events<br><extra></extra>", "<b>%{label}</b><br><extra></extra>"]
        )])

        macro_f1_score_fig = go.Figure(data=[go.Pie(
        labels=["Macro F1 Score", "Error Rate"],
        values=[macro_avg_metrics.get('f1-score', 0) * 100, 100 - macro_avg_metrics.get('f1-score', 0) * 100],
        hole=0.4,
        pull=[0, 0.1],
        hovertemplate=["<b>%{label}</b><br>F1 Score: Harmonic mean of Precision and Recall<br><extra></extra>", "<b>%{label}</b><br><extra></extra>"]
        )])

        execution_time_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=execution_time,
        gauge={
            'axis': {'range': [0, 300]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 100], 'color': "green"},
                {'range': [100, 200], 'color': "yellow"},
                {'range': [200, 300], 'color': "red"}
                ],
            }
        ))

        chart1_html = pio.to_html(class_distribution_fig, full_html=False)
        chart2_html = pio.to_html(accuracy_fig, full_html=False)
        chart3_html = pio.to_html(macro_precision_fig, full_html=False)
        chart4_html = pio.to_html(macro_recall_fig, full_html=False)
        chart5_html = pio.to_html(macro_f1_score_fig, full_html=False)
        execution_time_gauge_html = pio.to_html(execution_time_gauge, full_html=False)

        return render_template("monitor.html", metrics=metrics, chart1_html=chart1_html, chart2_html=chart2_html, chart3_html=chart3_html, chart4_html=chart4_html, chart5_html=chart5_html, execution_time_gauge_html=execution_time_gauge_html)
    return render_template("monitor.html", metrics={}, chart1_html="", chart2_html="", chart3_html="", chart4_html="", execution_time_gauge_html="")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)