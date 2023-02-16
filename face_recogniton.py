import cv2
import numpy as np

people=['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']
haar_cascade=cv2.CascadeClassifier('harcascasde/haarcascade_face.xml')

# features=np.load('features.npy')
# labels=np.load('labels.npy')

face_recognizer= cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img =cv2.imread(r'C:\Users\hp\Desktop\OPEN CV\Resources\Faces\val\jerry_seinfeld\3.jpg')

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('peron',gray)

        #DETECTING FACE
faces_rect=haar_cascade.detectMultiScale(gray,1.1,3)

for x,y,w,h in faces_rect:
    faces_roi=gray[y:y+h,x:x+h]
    
    label,confidence=face_recognizer.predict(faces_roi)
    print(f"label={people[label]} confidence={confidence}")
    cv2.putText(img,str(people[label]),(20,20),cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,0),2)
    cv2.rectangle(img,(x,y),(x+w,y+h),255,2) 
cv2.imshow('detected face',img)
cv2.waitKey(0)