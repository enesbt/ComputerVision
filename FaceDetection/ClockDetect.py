import cv2


img = cv2.imread("./Images/clock.jpg")

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

detection = cv2.CascadeClassifier("./Cascades/clocks.xml")

detections = detection.detectMultiScale(img,scaleFactor=1.03,minNeighbors=1)

for (x,y,w,h) in detections:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow("clock",img)
cv2.waitKey(0)
cv2.destroyAllWindows()







