import os
from PIL import Image

#pathforimages = "images\"
#listofimages = os.listdir(pathforimages)

with Image.open("images\image1.jpg") as image1:
	image1.rotate(45).show()
