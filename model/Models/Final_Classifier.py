#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from skimage.color import rgba2rgb
import pickle
import time


# In[10]:

start_time=time.time()

input_dir = '../../carla_lights/traffic_light_data'
categories = ['green', 'red', 'yellow']
blurred_suffix = ' blurred'  
img_size = (15, 15)  


data_train = []
labels_train = []
data_val = []
labels_val = []


# In[11]:


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


for category in categories:
    load_images(category, 'train', data_train, labels_train)


for category in categories:
    load_images(category, 'val', data_val, labels_val)


x_train = np.array(data_train)
y_train = np.array(labels_train)
x_val = np.array(data_val)
y_val = np.array(labels_val)


print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")


# In[12]:


## training SVC model with hyperparameter tuning
classifier = SVC(probability=True)  
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, parameters, cv=3)
grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

## Evaluation
y_pred = best_estimator.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

## Classification report
print(classification_report(y_val, y_pred))


# In[13]:


def classify_image_with_unknown(image_path, model, threshold=0.7):
   
    img = imread(image_path)
    ##RGB conversion
    if img.shape[-1] == 4:  
        img = rgba2rgb(img)
    img = resize(img, img_size).flatten()
    img = img.reshape(1, -1)  

    ## Prediction
    probabilities = model.predict_proba(img)
    max_confidence = np.max(probabilities)
    predicted_class = model.predict(img)[0]
    if max_confidence < threshold:
        return "unknown"
    return predicted_class


# In[14]:


model_file_name = '../../classifier'  # Specify the filename
with open(model_file_name, 'wb') as file:
    pickle.dump(best_estimator, file)

print(f"Model saved as {model_file_name}")


# In[16]:


# Path to the test image
test_image_path = "../../Messenger_creation_9203779869632344.jpg"

# Classify the image
result = classify_image_with_unknown(test_image_path, best_estimator, threshold=0.7)

# Print the result
print(f"Classification Result: {result}")

end_time = time.time()

#Metrics to be streamed
execution_time = end_time - start_time

# In[ ]:





# In[ ]:




