# Malicious Sign Detection

With the rise of autonomous vehicles, ensuring road safety through accurate traffic signal recognition has become a critical challenge. This project focuses on detecting and identifying malicious or altered traffic lights that could mislead self-driving cars, potentially causing accidents or unsafe driving behavior.

## 𝐎𝐛𝐣𝐞𝐜𝐭𝐢𝐯𝐞:
The project aimed to develop a robust system to detect and classify malicious traffic lights using custom-built machine learning models.

## 𝐌𝐞𝐭𝐡𝐨𝐝𝐨𝐥𝐨𝐠𝐲:
- Captured video feed from a camera to simulate real-world driving scenarios.
- Designed and implemented a custom machine learning model for traffic light detection and classification.
- Designed a classifier to process video frames and detect anomalies in light colors and patterns.
- Trained and evaluated the model using labeled datasets, focusing on performance metrics such as accuracy, precision, recall, and F1-score.
- Developed a Flask web application to monitor execution time, display performance metrics, and provide insights into model performance and behavior.

## 𝐑𝐞𝐬𝐮𝐥𝐭𝐬:
Achieved 95% accuracy in identifying tampered and malicious traffic lights under various conditions.

## 𝐂𝐨𝐧𝐜𝐥𝐮𝐬𝐢𝐨𝐧:
This project demonstrated the effectiveness of a custom machine learning model combined with computer vision techniques in enhancing the safety of autonomous vehicles by identifying and mitigating threats posed by malicious traffic lights. 
Work is ongoing to further improve the model's performance under challenging conditions, such as low light and fog.

## Technologies 
- **Python**: Programming language used for the implementation.
- **OpenCV** and **YOLO**: Library used for object classification and detection.
- **scikit-learn**: Library used for training and classification of images.
- **matplotlib**: Library for creating interactive visualizations.
- **Flask**: Web framework used to develop Monitoring Dashboard.
