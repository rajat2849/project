#Collect Data Sample from WebCam


import cv2
import numpy as np

#Face_Detect and Collect Sample

face_classifier = cv2.CascadeClassifier('D:/python/Project(Face Detection & Reconization)/Facial_Reconization(Single_Data)/opencv/haarcascades/haarcascade_frontalface_default.xml')
#Function which Detect Face

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Convert Image into Gray
    faces = face_classifier.detectMultiScale(gray,1.3,5) #Rectangle on Face

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w] #Size of Rectangle

    return cropped_face

#Open Web_cam

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

       #File Location where Sample Images Save

        file_name_path = 'D:/python/Project(Face Detection & Reconization)/Facial_Reconization(Single_Data)/Images/Rajat/'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)

        #Write text infront of Images

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Sample',face)
    else:
        print("Face not found")
        pass
       
    #Face Sample wait for 100 counting or stop when we hit "Enter" Key ==13

    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()
print('Samples Colletion Completed ')