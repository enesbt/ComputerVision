import cv2

img = cv2.imread("./Images/people3.jpg")
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detection = cv2.CascadeClassifier("./Cascades/fullbody.xml")

detections = detection.detectMultiScale(gray_image,scaleFactor=1.06,minNeighbors=7)

for (x,y,w,h) in detections:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


cv2.imshow("people",img)
cv2.waitKey(0)
cv2.destroyAllWindows()







