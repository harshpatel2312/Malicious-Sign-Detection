import os
from PIL import Image

pathforimages = "raw_images\\"
listofimages = os.listdir(pathforimages)

for eachimage in listofimages:
    image=Image.open(pathforimages+eachimage)
    resizedimage=image.resize((40,40))
    resizedimage.save(f"preprocessed_images\\resized_{eachimage}")
    
    
    
    