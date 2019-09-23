import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier(r"C:\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
labels = {"person-name": 1}
with open ("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
while(True):

    ret, frame = cap.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_,conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 255, 0)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)



        img_itm = '7.jpg'
        cv2.imwrite(img_itm, roi_color)
        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)



    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()