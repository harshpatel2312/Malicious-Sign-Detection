#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np


# In[7]:


def Detect_Circles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp = 1, minDist = 20, param1 = 50, param2 = 50, minRadius = 10, maxRadius = 50)

    return np.uint16(np.around(circles)) if circles is not None else None

