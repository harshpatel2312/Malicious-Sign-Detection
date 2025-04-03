import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.color import rgba2rgb, gray2rgb
import time
from flask import Flask, jsonify
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from concurrent.futures import ThreadPoolExecutor
import logging


def load_images(category, folder_type, data, labels):
    category_path = os.path.join(input_dir, folder_type, category)
    blurred_path = os.path.join(input_dir, folder_type, f"{category}{blurred_suffix}")

    paths = [category_path, blurred_path]
    image_files = []

    # Collecting all image file paths
    for folder in paths:
        if os.path.exists(folder):
            image_files.extend([os.path.join(folder, file) for file in os.listdir(folder)])

    def process_image(img_path):
        try:
            img = imread(img_path)
            # Converting grayscale and RGBA to RGB
            if len(img.shape) == 2:  # Grayscale
                img = gray2rgb(img)
            elif img.shape[-1] == 4:  # RGBA
                img = rgba2rgb(img)
            
            img = resize(img, img_size, anti_aliasing=True).flatten()
            return img, category
        except Exception as e:
            logging.warning(f"Skipping {img_path}: {e}")
            return None

    # Loading images using threading
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, image_files))

    # Adding valid images to dataset
    for res in results:
        if res:
            data.append(res[0])
            labels.append(res[1])

def to_numpy(data_train, labels_train, data_val, labels_val):
    global y_val
    x_train = np.array(data_train)
    y_train = np.array(labels_train)
    x_val = np.array(data_val)
    y_val = np.array(labels_val)
    
    logging.info(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")  
    return x_train, y_train, x_val, y_val

def plot_sample_images(x, y, categories, title="Sample Images from Dataset"):
    fig, axes = plt.subplots(1, len(categories), figsize=(12, 4))
    fig.suptitle(title, fontsize=14)
    
    for i, category in enumerate(categories):
        idx = np.where(y == category)[0][0]
        img = x[idx].reshape(img_size[0], img_size[1], 3)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(category)
        
    plt.show()

def train_svc(x_train, y_train, x_val, y_val):
    global y_pred
    classifier = SVC(probability=True)
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100]}]
    #parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    #parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # Improved cross-validation
    
    grid_search = GridSearchCV(classifier, parameters, cv=skf, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)
    
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    
    best_estimator = grid_search.best_estimator_
    
    y_pred = best_estimator.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    logging.info(f"Validation Accuracy: {accuracy * 100:.2f}%")

    report = classification_report(y_val, y_pred, output_dict=True)
    with open('Resources/classification_report', 'wb') as f:
        pickle.dump(report, f)
    print(report)
    
    return y_pred, best_estimator, report

def conf_matrix(y_val, y_pred, categories):
    cm = confusion_matrix(y_val, y_pred, labels=categories)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
input_dir = r"E:\Education\Projects\Machine Learning\Computer Vision\Malicious-Sign-Detection\traffic_light_data"
categories = ['green', 'red', 'yellow']
blurred_suffix = ' blurred'  
img_size = (15, 15)

data_train = []
labels_train = []
data_val = []
labels_val = []

for category in categories:
    load_images(category, 'train', data_train, labels_train)
for category in categories:
    load_images(category, 'val', data_val, labels_val)

# Model Training 
x_train, y_train, x_val, y_val = to_numpy(data_train, labels_train, data_val, labels_val)

# Plot images from dataset
plot_sample_images(x_train, y_train, categories)

# Training SVC with hyper parameter tuning
y_pred, best_estimator, report= train_svc(x_train, y_train, x_val, y_val)

#C Confusion Matrix
conf_matrix(y_val, y_pred, categories)

# Pickle File
model_file_name = r"E:\Education\Projects\Machine Learning\Computer Vision\Malicious-Sign-Detection\model\Models\classifier"  # Specify the filename
with open(model_file_name, 'wb') as file:
    pickle.dump(best_estimator, file)
logging.info(f"Model saved as {model_file_name}")
