import cv2
import numpy as np
from os import listdir #import  file from loaction
from os.path import isfile, join #import path

data_path = 'D:/python/Project(Face Detection & Reconization)/Facial_Reconization(Single_Data)/Images/Rajat/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))] #connect file

Training_Data, Labels = [], [] #martix store in both array

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

 
#Local Binary face algoritham 
#LBPH is one of the easiest face recognition algorithms.
# It can represent local features in the images. 
# It is possible to get great results (mainly in a controlled environment). 
# It is robust against monotonic gray scale transformations. 
# It is provided by the OpenCV library (Open Source Computer Vision Library).
model = cv2.face.LBPHFaceRecognizer_create()  #Store Binary code of Images




model.train(np.asarray(Training_Data), np.asarray(Labels)) #call functions

print("Dataset Model Training Completed ")
