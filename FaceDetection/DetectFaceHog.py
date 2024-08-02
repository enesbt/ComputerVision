import dlib
import cv2


img = cv2.imread("./Images/people2.jpg")

"""
face_detector_hog = dlib.get_frontal_face_detector()

detections = face_detector_hog(img,1)

for face in detections:
    l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
    cv2.rectangle(img,(l,t),(r,b),(0,255,0),2)

"""
face_detector_cnn = dlib.cnn_face_detection_model_v1('./Weights/mmod_human_face_detector.dat')

detections = face_detector_cnn(img,1)

for face in detections:
    l,t,r,b,c = face.rect.left(),face.rect.top(),face.rect.right(),face.rect.bottom(),face.confidence
    cv2.rectangle(img,(l,t),(r,b),(0,255,0),2)



cv2.imshow("people",img)
cv2.waitKey(0)
cv2.destroyAllWindows()