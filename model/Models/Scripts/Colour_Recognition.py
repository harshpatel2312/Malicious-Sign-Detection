#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[41]:


#Setting Index values
index = ["color", "color_name", "hex", "R", "G", "B"]
df = pd.read_csv("E:/Education/Projects/Machine Learning/Computer Vision/MSD Models/Resources/Files/colors.csv", 
                 names = index, header = None)


# In[43]:


def Recognize_Color(R, G, B):
    minimum = 10000
    for i in range(len(df)):
        #Manhattan Distance Formula (Calculates the difference in the actual and csv RGB color values)
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = df.loc[i, "color_name"]
    return cname

