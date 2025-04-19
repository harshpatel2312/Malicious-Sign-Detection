# 🧠 Train Traffic Light Classification Model

This branch contains everything you need to **train your own SVM-based traffic light classifier** from scratch using image data categorized by light color.

---

## 📁 Required File Structure
Make sure your training data is structured like this:
```bash
Train_Model
├── model/
|   ├── Scripts/
|       ├── Defence_Train.py
|       ├── Resources/
├── traffic_light_data/ # Your image dataset should follow this structure
|   ├── train/
|        ├── red/
|        ├── green/
|        └── yellow/
|   ├── val/
|       ├── red/
|       ├── green/
|       └── yellow/
```

- The `train/` and `val/` folders must contain images categorized into folders named after their class (`red`, `yellow`, `green`).
- You can optionally include blurred images inside folders named like `"red blurred"`.

---

## 🔄 Clone the Repository & Switch to Branch
```bash
git clone https://github.com/harshpatel2312/Malicious-Sign-Detection.git
cd Malicious-Sign-Detection
git checkout train_model
```

---

## 🧰 Setting Up the Environment
1. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
2. Install required packages
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Training Script
Navigate to the script directory and run:
```bash
cd model/Scripts
python Defence_Train.py
```

---

## 💾 Output
* A trained SVM model will be saved as `model/Trained_with_threshold.pkl`.
* Classification report (serialized with pickle) will be saved as `model/Scripts/Resources/classification_report`.
* Console output includes::
  * Best hyperparameters
  * Accuracy with 95% confidence interval
  * Confusion matrix
  * Confidence score distribution
  * Sample misclassified entries

---

## 🧪 Evaluation Metrics
The script uses:
  * 5-fold Stratified Cross Validation
  * Class-specific confidence thresholds
  * Confusion matrix & classification report
  * Confidence distribution plots
  * Misclassification analysis

---

## 📦 Customize
You can change:
  * `img_size = (20, 20)` — if training on higher-res features
  * `thresholds = {...}` — adjust confidence thresholds per class
  * `categories = ['red', 'yellow', 'green']` — add more labels if needed

---

## 🧠 Use the Model in `Traffic_Light_Presence_Detector.py`
Once trained, the model can be used for prediction in Traffic_Light_Presence_Detector.py (on the [`detection_model`](https://github.com/harshpatel2312/Malicious-Sign-Detection/tree/detection_model) branch).
Visit [`detection_model`](https://github.com/harshpatel2312/Malicious-Sign-Detection/tree/detection_model) branch for more details.

---

## 📃 License
MIT License © Harsh Patel
