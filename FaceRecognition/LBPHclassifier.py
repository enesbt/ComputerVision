import numpy as np
import cv2
from PIL import Image
import os
from sklearn.metrics import  accuracy_score



def get_image_data():
    paths = [os.path.join('./yalefaces/train',f) for f in os.listdir('./yalefaces/train')]

    faces = []
    ids = []
    for path in paths:
        image = Image.open(path).convert('L')
        image_np = np.array(image,'uint8')
        id = os.path.split(path)[1].split('.')[0].replace('subject','')
        ids.append(id)
        faces.append(image_np)
    return  np.array(ids),faces



ids,faces = get_image_data()
ids = np.array(ids,dtype=np.int32)

#lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
#lbph_classifier.train(faces,ids)
#lbph_classifier.write('lbph_classifier.yml')


"""

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.read('./lbph_classifier.yml')


test_image = './yalefaces/test/subject01.happy.gif'
image = Image.open(test_image).convert('L')
image_np = np.array(image,'uint8')

prediction = lbph_classifier.predict(image_np)
print(prediction)

expected_output = int(os.path.split(test_image)[1].split('.')[0].replace('subject',''))
print(expected_output)


cv2.putText(image_np,'Pred: '+str(prediction[0]),(20,30),1,1,cv2.FONT_HERSHEY_SIMPLEX)
cv2.putText(image_np,'Exp: '+str(expected_output),(20,60),1,1,cv2.FONT_HERSHEY_SIMPLEX)

"""

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.read('./lbph_classifier.yml')

paths = [os.path.join('./yalefaces/train',f) for f in os.listdir('./yalefaces/train')]
predictions = []
expected_outputs = []

for path in paths:
    image = Image.open(path).convert('L')
    image_np = np.array(image, 'uint8')
    prediction = lbph_classifier.predict(image_np)
    expected_output = int(os.path.split(path)[1].split('.')[0].replace('subject',''))


    predictions.append(prediction)
    expected_outputs.append(expected_output)

predictions = np.array(predictions)
expected_outputs = np.array(expected_outputs)
acc = accuracy_score(expected_outputs,predictions)
print(acc)

#cv2.imshow('image',image_np)
cv2.waitKey(0)
cv2.destroyAllWindows()

