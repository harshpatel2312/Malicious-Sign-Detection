import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.color import rgba2rgb
import time
from flask import Flask, jsonify
import threading
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def load_images(category, folder_type, data, labels):
    category_path = os.path.join(input_dir, folder_type, category)
    blurred_path = os.path.join(input_dir, folder_type, f"{category}{blurred_suffix}")
    for folder in [category_path, blurred_path]:
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            try:
                img = imread(img_path)
                #Converting RGBA to RGB if necessary
                if img.shape[-1] == 4:  # If image has 4 channels (RGBA)
                    img = rgba2rgb(img)
                img = resize(img, img_size)
                if img.shape == (15, 15, 3):  
                    data.append(img.flatten())
                    labels.append(category)
                else:
                    print(f"Skipping {file}: Invalid shape after resize {img.shape}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

def train(data_train, labels_train, data_val, labels_val):
    x_train = np.array(data_train)
    y_train = np.array(labels_train)
    x_val = np.array(data_val)
    y_val = np.array(labels_val)
    
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")  
    return x_train, y_train, x_val, y_val

def hyper_tuning(x_train, y_train, x_val, y_val):
    classifier = SVC(probability=True)
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    #parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    
    grid_search = GridSearchCV(classifier, parameters, cv=3)
    grid_search.fit(x_train, y_train)
    
    best_estimator = grid_search.best_estimator_
    
    y_pred = best_estimator.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_val, y_pred))  
    return y_val, y_pred, best_estimator

def conf_matrix(y_val, y_pred, categories):
    cm = confusion_matrix(y_val, y_pred, labels=categories)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    input_dir = r"E:\Education\Projects\Machine Learning\Computer Vision\Malicious-Sign-Detection\traffic_light_data"
    categories = ['green', 'red', 'yellow', "unknown"]
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
    x_train, y_train, x_val, y_val = train(data_train, labels_train, data_val, labels_val)
    
    # Hyper Parameter Tuning
    y_val, y_pred, best_estimator = hyper_tuning(x_train, y_train, x_val, y_val)

    #C Confusion Matrix
    conf_matrix(y_val, y_pred, categories)

    # Pickle File
    model_file_name = r"E:\Education\Projects\Machine Learning\Computer Vision\Malicious-Sign-Detection\model\Models\classifier"  # Specify the filename
    with open(model_file_name, 'wb') as file:
        pickle.dump(best_estimator, file)
    print(f"Model saved as {model_file_name}")
    