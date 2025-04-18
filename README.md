# 🌐 Traffic Light Monitoring Dashboard

This branch provides a lightweight **Flask web application** for monitoring the classification results of a traffic light detection system. It is designed to run in tandem with `Traffic_Light_Presence_Detector.py`, displaying model metrics and predictions on a browser interface.

---

## 📁 Project Structure
```bash
Web_Monitoring/
├── static/
│   └── monitor.css         # CSS for styling the dashboard
├── templates/
│   ├── root.html
│   ├── home.html
│   └── monitor.html        # Web dashboard page
├── .gitignore              # Ignore virtual envs, logs, etc.
├── app.py                  # Flask app that serves metrics to the dashboard
├── requirements.txt        # Dependencies
```

---

## ⚙️ Setup Instructions

### 1. Clone This Branch

```bash
git clone -b web_monitoring https://github.com/harshpatel2312/Malicious-Sign-Detection.git
cd Malicious-Sign-Detection/Web_Monitoring
```
### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Running the Web Dashboard
Start the Flask dashboard:
```python
python app.py
```
Then visit: 
🔗 http://127.0.0.1:5001/monitor

## 🧪 What You’ll See
* Execution time
* Per-frame predictions streamed live to the dashboard

## 🧾 Notes
* The dashboard refreshes every 1 second.
* This app only displays classification results — all detection and classification is handled in the backend by the `Traffic_Light_Presence_Detector.py`.
* For integration help with the detection system, refer to the `detection_model` branch.

## 📃 License
MIT License © Harsh Patel
