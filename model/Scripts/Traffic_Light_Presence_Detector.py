import cv2
from screeninfo import get_monitors
import pickle
import time
from ultralytics import YOLO
from classify import test
from flask import Flask, jsonify
import threading

# Flask Service
app = Flask(__name__)

def run_flask():
    app.run(host='127.0.0.1', port=5000, debug=False)

# Loading the trained model
model_file_name = r"..\Trained_with_threshold.pkl"
with open(model_file_name, 'rb') as file:
    best_estimator = pickle.load(file)

# Loading classification_report
with open('Resources/classification_report', 'rb') as f:
    report = pickle.load(f)
report["Execution_Time_of_Prediction"] = None
report["Test_Results"] = []

@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(report), 200

threading.Thread(target=run_flask, daemon=True).start()

img_size = (20, 20) # Resizing for consistency

# Get screen resolution
screen = get_monitors()[0]  # Get the primary monitor
screen_width, screen_height = screen.width, screen.height

# Load YOLO model for traffic light detection
model = YOLO('yolov8n.pt') 

# Load the video
video_path = r"Resources\Videos\Real_life_test_daylight_3.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Cannot open video file!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  # Create a resizable window
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or cannot access the video.")
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_count += 1
    # Process every frame or skip to reduce processing
    if frame_count % int(fps) != 0:  # Process one frame per second
        continue

    # YOLO detection
    results = model(frame)  # Run the YOLO model on the current frame
    
    # Access the first result in the list
    result = results[0]

    test_results = {}

    # Iterate through detected boxes
    for box in result.boxes:
        class_id = int(box.cls)  # Object class ID
        if class_id == 9:  # class 9 corresponds to traffic lights in YOLO
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            traffic_light_roi = frame[y1:y2, x1:x2]  # Crop the traffic light region

            test_results[f"frame_{frame_count}"] = test(traffic_light_roi, img_size, frame_count, best_estimator)
            report["Test_Results"].append(test_results)

            break  # Avoid multiple saves for the same frame
            
    # Resize frame to fit screen size
    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = frame_width / frame_height

    # Calculate new dimensions while maintaining aspect ratio
    if frame_width > screen_width or frame_height > screen_height:
        if frame_width / screen_width > frame_height / screen_height:
            new_width = screen_width
            new_height = int(screen_width / aspect_ratio)
        else:
            new_height = screen_height
            new_width = int(screen_height * aspect_ratio)
    else:
        new_width, new_height = frame_width, frame_height

    resized_frame = cv2.resize(frame, (new_width, new_height))
    cv2.imshow("Video", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

end_time = time.time()
execution_time = end_time - start_time
report["Execution_Time_of_Prediction"] = execution_time
print(f"Execution Time: {execution_time}")

print(f"Total frames processed: {frame_count}")
cap.release()
cv2.destroyAllWindows()

