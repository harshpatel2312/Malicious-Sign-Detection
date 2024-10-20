from flask import Flask

app=Flask(__name__)

@app.route("/")
def render_home():
    return "<h1 style='color:red; text-align:center;'>Render Home Page here</h1>"

@app.route("/monitor")
def render_monitor():
    return "<h1 style='color:red; text-align:center;' >Render Monitor Page here</h1>"

@app.route("/about")
def render_about():
    return "<h1 style='color:red; text-align:center;'>Render About Page here</h1>"