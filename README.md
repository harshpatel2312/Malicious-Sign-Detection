## 🚦 Traffic Light Presence Detection

`Traffic_Light_Presence_Detector.py` detects and classifies traffic light colors in video frames using a combination of **YOLOv8** object detection and a custom-trained **SVM classifier**.

---

### 📁 Required File Structure

Ensure the following file structure is maintained:

```bash
Malicious-Sign-Detection/
│
├── model/
│   ├── Trained_with_threshold.pkl
│   └── Scripts/
|       ├── Resources/
|           ├── Videos/
|               ├── your_video.mp4
|           ├── classification_report.pkl
│       ├── classify.py
|       ├── Defence_Train.py
│       └── Traffic_Light_Presence_Detector.py
├── requirements.txt
```

### 🔄 Option 1: Clone the Repository

```bash
git clone https://github.com/harshpatel2312/Malicious-Sign-Detection.git
cd Malicious-Sign-Detection
```

### 🧰 Setting Up Environment & Installing Dependencies
1. **Create a virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
2. **Install required packages**
```bash
pip install -r requirements.txt
```

### 📥 Option 2: Download Individual Files
If you prefer to download only a few files, ensure the following are placed together in the correct structure:
* `Traffic_Light_Presence_Detector.py`
* `classify.py`
* `Trained_with_threshold.pkl`

You must then update the file paths in `Traffic_Light_Presence_Detector.py` manually. For example:
```python
from classify import test # Modify the path to `classify.py`, if not using predefined file structure
model_file_name = "Trained_with_threshold.pkl" # Modify path as necessary
```

### 🎥 Using Your Own Video
To test the model with your own video:
1. Place your video file inside the `model/Scripts/Resources/Videos/` folder.
2. Update the path in `Traffic_Light_Presence_Detector.py`:
```python
video_path = r"Scripts/Resources/Videos/your_video.mp4" # Modify path as necessary
```

### 🧪 Running Traffic Light Presence Detector
Navigate to the `Scripts` directory and run:
```bash
cd Malicious-Sign-Detection/model/Scripts
python Traffic_Light_Presence_Detector.py
```

### 📄 Output
* Displays the video in `greyscale`.
* Prints the predicted light color (red, yellow, green, or unknown) per frame.

### 💼 Use in Your Own Projects
You are welcome to integrate this classifier into your own applications. Simply:
* Import `classify.py` and the trained `Trained_with_threshold.pkl` model.
* Resize and preprocess your traffic light images or frames as shown.
* Call the `classify_image_with_unknown()` function.

This modular design allows easy reuse in smart city systems, autonomous driving, or surveillance analysis.

### 🧠 YOLOv8 Weights
Make sure you have `yolov8n.pt` downloaded or accessible if not already present in your environment. You can download it from: https://github.com/ultralytics/ultralytics
