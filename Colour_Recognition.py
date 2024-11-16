import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, jsonify
import requests
import threading

app = Flask(__name__)

def run_flask():
    app.run(debug=False, port=5001)

recognized_colors = []

@app.route('/color', methods=['POST'])
def post_color():
    data = request.json
    recognized_colors.append(data) 
    return jsonify({"message": "Color data received!"}), 200

@app.route('/color', methods=['GET'])
def get_color():
    return jsonify(recognized_colors), 200

img = cv2.imread("test_image.jpg")
index = ["color", "color_name", "hex", "R", "G", "B"]
df = pd.read_csv("colors.csv", names = index, header = None)
clicked = False
r = g = b = xpos = ypos = 0

def recognize_color(R, G, B):
    minimum = 10000
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = df.loc[i, "color_name"]
    return cname


def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

cv2.namedWindow("Color Recognition")
cv2.setMouseCallback("Color Recognition", mouse_click)

if __name__ == '__main__':
    threading.Thread(target=run_flask, daemon=True).start()

    with open("Retrieved Colors.txt", "w") as f:
        while True:
            cv2.imshow("Color Recognition", img)
            if clicked:
                color_name = recognize_color(r, g, b)
                text = f"{color_name} R = {r} G = {g} B = {b}"
                f.write(text + "\n")
                print(text)
                
                rgb_string = f"{r},{g},{b}"
                requests.post('http://127.0.0.1:5000/color', json={
                    "color_info": f"{color_name} ({rgb_string})"
                })

                clicked = False
            if cv2.waitKey(20) & 0xFF == 27:
                break