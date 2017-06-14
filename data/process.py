import cv2
import os

files=os.listdir()

for file in files:
	print(file)
	if(file.endswith(".tif")):
		file_name=file.split("=")[1]
		image=cv2.imread(file)
		height,width,_=image.shape
		image=cv2.resize(image,(width,50))
		label=file_name.split(".")[0]
		if(len(label)==9):
			cv2.imwrite("./ok/"+file_name,image)
		# if(len(label)==7):
			# file_name=file_name[:1]+"-"+file_name[2:4]+"-"+file_name[5:6]
			# cv2.imwrite("./ok/"+file_name+".tiff",image)