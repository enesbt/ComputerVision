import cv2


cap = cv2.VideoCapture(0)

while True:
    res,frame = cap.read()

    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier('./Cascades/haarcascade_frontalface_default.xml')
    detections = detector.detectMultiScale(gray_frame)
    for (x,y,w,h) in detections:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

