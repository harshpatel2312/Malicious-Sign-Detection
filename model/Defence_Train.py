import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.color import rgba2rgb, gray2rgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
import math


# Make paths relative to the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
input_dir = os.path.join(ROOT_DIR, "traffic_light_data")
model_output = os.path.join(ROOT_DIR, "model", "Trained_with_threshold.pkl")
report_output = os.path.join(ROOT_DIR, "model", "Scripts", "Resources", "classification_report")


#input_dir = r"traffic_light_data"
categories = ['green', 'red', 'yellow']
blurred_suffix = ' blurred'
img_size = (20, 20) 
thresholds = {
    'green': 0.52,   
    'red': 0.62,     
    'yellow': 0.72   
}
threshold = 0.7

    
#model_output = r"model\Trained_with_threshold.pkl"
#report_output = r"model\Scripts\Resources\classification_report"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_images(category, folder_type, data, labels):
    category_path = os.path.join(input_dir, folder_type, category)
    blurred_path = os.path.join(input_dir, folder_type, f"{category}{blurred_suffix}")
    paths = [category_path, blurred_path]
    image_files = []

    for folder in paths:
        if os.path.exists(folder):
            image_files.extend([os.path.join(folder, file) for file in os.listdir(folder)])

    def process_image(img_path):
        try:
            img = imread(img_path)
            if len(img.shape) == 2:
                img = gray2rgb(img)
            elif img.shape[-1] == 4:
                img = rgba2rgb(img)
            img = resize(img, img_size, anti_aliasing=True).flatten()
            return img, category
        except Exception as e:
            logging.warning(f"Skipping {img_path}: {e}")
            return None

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, image_files))

    for res in results:
        if res:
            data.append(res[0])
            labels.append(res[1])

def to_numpy(data_train, labels_train, data_val, labels_val):
    x_train = np.array(data_train)
    y_train = np.array(labels_train)
    x_val = np.array(data_val)
    y_val = np.array(labels_val)
    return x_train, y_train, x_val, y_val

def train_svc(x_train, y_train):
    classifier = SVC(probability=True)
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 500, 1000]}]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(classifier, parameters, cv=skf, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def predict_with_class_thresholds(model, x_val, thresholds):
    probs = model.predict_proba(x_val)
    preds = model.predict(x_val)
    labels = model.classes_
    
    final_preds = []
    confidences = []
    
    for i in range(len(x_val)):
        pred_label = preds[i]
        confidence = np.max(probs[i])
        threshold = thresholds.get(pred_label, 0.7)  # default fallback
        
        if confidence >= threshold:
            final_preds.append(pred_label)
        else:
            final_preds.append("unknown")
        
        confidences.append(confidence)
    
    return final_preds, confidences


def evaluate(y_val, y_pred, labels):
    print("\nClassification Report:")
    report = classification_report(y_val, y_pred, labels=labels + ['unknown'], output_dict=False)
    print(report)
    cm = confusion_matrix(y_val, y_pred, labels=labels + ['unknown'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels + ['unknown'], yticklabels=labels + ['unknown'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


data_train, labels_train, data_val, labels_val = [], [], [], []

for category in categories:
    load_images(category, 'train', data_train, labels_train)
    load_images(category, 'val', data_val, labels_val)

x_train, y_train, x_val, y_val = to_numpy(data_train, labels_train, data_val, labels_val)

# Train and predict
best_estimator = train_svc(x_train, y_train)

y_pred, confidences = predict_with_class_thresholds(best_estimator, x_val, thresholds)


# Evaluate
evaluate(y_val, y_pred, categories)

# Save model
with open(model_output, 'wb') as file:
    pickle.dump(best_estimator, file)
logging.info(f"Model saved as {model_output}")

# Predict with probability
probs = best_estimator.predict_proba(x_val)
pred_labels = best_estimator.predict(x_val)
max_confidences = np.max(probs, axis=1)
final_preds = np.where(max_confidences < threshold, 'unknown', pred_labels)

# Classification report
report = classification_report(y_val, final_preds, labels=categories + ['unknown'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("\n Classification Report:\n")
print(report_df)

#  Confusion Matrix
cm = confusion_matrix(y_val, final_preds, labels=categories + ['unknown'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories + ['unknown'], yticklabels=categories + ['unknown'])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Confidence Distribution Plot
plt.figure(figsize=(8, 5))
sns.histplot(max_confidences, bins=20, kde=True, color='teal')
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
plt.title("Distribution of Prediction Confidences")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Step 5: Confidence vs Actual Comparison
confidence_df = pd.DataFrame({
    'True_Label': y_val,
    'Predicted_Label': final_preds,
    'Confidence': max_confidences
})

# Show misclassified samples
misclassified = confidence_df[confidence_df['True_Label'] != confidence_df['Predicted_Label']]
print(f"\n Total Misclassifications: {len(misclassified)} / {len(y_val)}")
print("\n Sample Misclassified Entries:\n", misclassified.head())

y_pred = best_estimator.predict(x_val)
report_dict = classification_report(y_val, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print(report_df)


accuracy = accuracy_score(y_val, y_pred)
n = len(y_val)
z = 1.96  # 95% confidence

stderr = math.sqrt((accuracy * (1 - accuracy)) / n)
margin = z * stderr

print(f"Accuracy: {accuracy:.4f} ± {margin:.4f} (95% CI)")

y_probs = best_estimator.predict_proba(x_val)
max_probs = np.max(y_probs, axis=1)

plt.hist(max_probs, bins=20, color='orange', edgecolor='black')
plt.title('Prediction Confidence Distribution')
plt.xlabel('Max Probability')
plt.ylabel('Frequency')
plt.show()
misclassified_idxs = np.where(y_pred != y_val)[0]
print(f"Total misclassified samples: {len(misclassified_idxs)}")

