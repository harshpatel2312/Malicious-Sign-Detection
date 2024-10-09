import os
from PIL import Image
import time

start_time=time.time()
print("Preprocessing has been started.")

pathforimages = "raw_images\\"
listofimages = os.listdir(pathforimages)

for eachimage in listofimages:
    image=Image.open(pathforimages+eachimage)
    resizedimage=image.resize((40,40))
    resizedimage.save(f"preprocessed_images\\resized_{eachimage}")

end_time=time.time()
print(f"Preprocessing has been completed in {end_time-start_time} seconds.")
    
    
    
    