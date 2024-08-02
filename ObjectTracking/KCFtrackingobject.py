import cv2

tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture("race.mp4")
ok,frame = video.read()

bbox = cv2.selectROI(frame)

ok = tracker.init(frame,bbox)


while True:
    ok,frame = video.read()
    if not ok:
        break
    ok,bbox = tracker.update(frame)

    if ok:
        (x,y,w,h) = [int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("tracking",frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break