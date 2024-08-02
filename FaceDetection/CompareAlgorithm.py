import cv2
import dlib


img = cv2.imread("./Images/people2.jpg")
img2 = img.copy()
img3 = img.copy()
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#haarcascade
cascade_detection = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')
detections = cascade_detection.detectMultiScale(gray_img)
for (x,y,w,h) in detections:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

#HOG
detection_hog = dlib.get_frontal_face_detector()
detections_hog = detection_hog(img2,1)
for face in detections_hog:
    cv2.rectangle(img2,(face.left(),face.top()),(face.right(),face.bottom()),(255,0,0),2)


#CNN
detection_cnn = dlib.cnn_face_detection_model_v1('./Weights/mmod_human_face_detector.dat')
detections_cnn = detection_cnn(img3,1)

for face in detections_cnn:
    cv2.rectangle(img2,(face.rect.left(),face.rect.top()),(face.rect.right(),face.rect.bottom()),(255,0,0),2)




cv2.imshow("haarcascade metod",img)
cv2.imshow("HOG metod",img2)
cv2.imshow("CNN metod",img3)

cv2.waitKey(0)
cv2.destroyAllWindows()