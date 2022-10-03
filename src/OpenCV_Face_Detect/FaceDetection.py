import cv2

# Read webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    # Convert into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces and crop the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        faces = image[y:y + h, x:x + w]

    if len(faces) > 0:
        faces = cv2.resize(faces, (100, 100))
        print(len(faces))
        cv2.imshow("face", cv2.flip(faces, 1))
    cv2.imshow('img', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()