import cv2

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('haarcascade_eye.xml')
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


while True:
    _, pic = camera.read()
    gray = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY)
    faces = face.detectMultiScale(gray, 1.1, 4)
    eyes = eye.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(pic, (x, y), (x + w, y + h), (255, 255, 0), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(pic, (x, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.imshow('Detected Face and eye', pic)

    q = cv2.waitKey(10)
    if q == ord("q"):
        break
camera.release()
