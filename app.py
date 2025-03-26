from flask import Flask, render_template
import requests
import plotly.graph_objects as go
import plotly.io as pio

app=Flask(__name__)

def get_metrics():
    try:
        response = requests.get('http://24.150.183.74:5000/data')
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
    precision_macro = metrics['macro avg']['precision']*100
    recall_macro = metrics['macro avg']['recall']*100
    f1_score_macro = metrics['macro avg']['f1-score']*100
    support_macro = metrics['macro avg']['support']
    support_red = metrics['red']['support']
    support_yellow = metrics['yellow']['support']
    support_green = metrics['green']['support']
    support_unknown = metrics['unknown']['support']

    fig1 = go.Figure(data=[go.Pie(
        labels=["Green", "Yellow", "Red", "Unknown"],
        values=[support_red, support_yellow, support_green, support_unknown],
        hole=0.4,
        pull=[0, 0, 0, 0.1],
        marker=dict(
        colors=["#00FF00", "#FFFF00", "#FF0000", "#A9A9A9"])
    )])
    fig1.update_layout(title_text="Macro Support")

    fig2 = go.Figure(data=[go.Pie(
        labels=["Macro Precision", "Error Rate"],
        values=[precision_macro, 100 - precision_macro],
        hole=0.4,
        pull=[0, 0.1],
        hovertemplate=["<b>%{label}</b><br>Refers to fraction of true positive among all the positives: TP/TP+FP<br><extra></extra>","<b>%{label}</b><br><extra></extra>"]
    )])
    fig2.update_layout(title_text="Macro Precision")

    fig3 = go.Figure(data=[go.Pie(
        labels=["Macro Recall", "Error Rate"],
        values=[recall_macro, 100 - recall_macro],
        hole=0.4,
        pull=[0, 0.1],
        hovertemplate=["<b>%{label}</b><br>Refers to fraction of true positive among all correct events: TP/TP+FN<br><extra></extra>","<b>%{label}</b><br><extra></extra>"]
    )])
    fig3.update_layout(title_text="Macro Recall")

    fig4 = go.Figure(data=[go.Pie(
        labels=["Macro F1 Score", "Error Rate"],
        values=[f1_score_macro, 100 - f1_score_macro],
        hole=0.4,
        pull=[0, 0.1],
        hovertemplate=["<b>%{label}</b><br>Refers to harmonic mean of Precision and Recall: 2 x (PrecisionxRecall)/(Precision+Recall)<br><extra></extra>","<b>%{label}</b><br><extra></extra>"]
    )])
    fig4.update_layout(title_text="Macro F1 Score")

    chart1_html = pio.to_html(fig1, full_html=False)
    chart2_html = pio.to_html(fig2, full_html=False)
    chart3_html = pio.to_html(fig3, full_html=False)
    chart4_html = pio.to_html(fig4, full_html=False)

    return render_template("monitor.html",
                           metrics=metrics,
                           chart1_html=chart1_html,
                           chart2_html=chart2_html,
                           chart3_html=chart3_html,
                           chart4_html=chart4_html)

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)