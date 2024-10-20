from flask import Flask

app=Flask(__name__)

@app.route("/")
def initial_test():
    return "<p>Hello World! This is the initial test</p>"