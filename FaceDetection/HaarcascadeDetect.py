import cv2
import numpy as np

img = cv2.imread("Images/people1.jpg")
print(img.shape)

#img = cv2.resize(img,(800,600))

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("gray shape",img_gray.shape)

face_detector = cv2.CascadeClassifier("./Cascades/haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("./Cascades/haarcascade_eye.xml")


detections = face_detector.detectMultiScale(img_gray,scaleFactor=1.3,minSize=(30,30))
eye_detections = eye_detector.detectMultiScale(img_gray,scaleFactor=1.1,minNeighbors=10,maxSize=(70,70))



for (x,y,w,h) in detections:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


for (x,y,w,h) in eye_detections:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("people",img)


img2 = cv2.imread("Images/people2.jpg")
print(img2.shape)

#img2 = cv2.resize(img2,(800,600))

img_gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
print("gray shape",img_gray2.shape)


detections2 = face_detector.detectMultiScale(img_gray2,scaleFactor=1.15,minNeighbors=7,minSize=(20,20),
                                             maxSize=(200,200))

#print(detections)

for (x,y,w,h) in detections2:
    cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("people2",img2)


cv2.waitKey(0)
cv2.destroyAllWindows()