import cv2
import numpy as np
import os
import math

class PatternRecognition:
    @staticmethod
    def patternRecognition():
        train_path = './pattern_recognition/train'
        names = os.listdir(train_path)
        face_cascade = cv2.CascadeClassifier('./pattern_recognition/haarcascade_frontalface_default.xml')

        face_list = []
        class_list = []

        for i, name in enumerate(names):
            full_path = os.path.join(train_path, name)

            for image_path in os.listdir(full_path):
                full_image_path = os.path.join(full_path, image_path)
                img_gray = cv2.imread(full_image_path, 0)
                detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.5, minNeighbors=7)

                if len(detected_face) < 1:
                    continue

                for face_rect in detected_face:
                    a, b, c, d = face_rect
                    face_img = img_gray[b:b+d, a:a+c]
                    face_img = cv2.resize(face_img, (200, 200))
                    face_list.append(face_img)
                    class_list.append(i)

        face_recog = cv2.face.LBPHFaceRecognizer_create()
        face_recog.train(face_list, np.array(class_list))

        test_path = './pattern_recognition/test'

        for image_path in os.listdir(test_path):
            full_path = os.path.join(test_path, image_path)
            img_bgr = cv2.imread(full_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            detected_face = face_cascade.detectMultiScale(img_gray, scaleFactor=1.5, minNeighbors=7)

            if len(detected_face) < 1:
                continue

            for face_rect in detected_face:
                a, b, c, d = face_rect
                face_img = img_gray[b:b+d, a:a+c]
                face_img = cv2.resize(face_img, (200, 200))
                res, conf = face_recog.predict(face_img)
                conf = math.floor(conf * 100) / 100

                cv2.rectangle(img_bgr, (a, b), (a+c, b+d), (0, 255, 0), 1)
                desc = names[res] + ' ' + str(conf) + '%'
                cv2.putText(img_bgr, desc, (a, b-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0), 1)
                cv2.imshow('res', img_bgr)
                cv2.waitKey(0)
            

        for image_path in os.listdir(test_path):
            full_path = os.path.join(test_path, image_path)
            img_bgr = cv2.imread(full_path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

            detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)

            for face_rect in detected_faces:
                x, y, w, h = face_rect
                face_region = img_gray[y:y+h, x:x+w]
                face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
                img_gray[y:y+h, x:x+w] = face_region

                cv2.imshow('Face Blur', img_gray)
                cv2.waitKey(0)

