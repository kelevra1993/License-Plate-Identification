import os 
import cv2
from random import shuffle

files=os.listdir("./valid")
shuffle(files)
shuffle(files)
file_counter=0
for file in files:
	if(file.endswith(".tif")):
		if(file_counter<900):
			os.rename("./valid/"+file,"./ok/"+file)
		file_counter=file_counter+1