import cv2
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(gray_image)

    if results.detections:
      h, w, _ = image.shape
      x1 = int(results.detections[0].location_data.relative_bounding_box.xmin * w)
      y1 = int(results.detections[0].location_data.relative_bounding_box.ymin * h)
      width = int(results.detections[0].location_data.relative_bounding_box.width * w)
      height = int(results.detections[0].location_data.relative_bounding_box.height * h)

      x2 = x1 + width
      y2 = y1 + height

      img_resize = cv2.resize(image[y1:y2, x1:x2, :], (100, 100))
      cv2.imshow('Cut', cv2.flip(img_resize, 1))

      cv2.imwrite('Scene.jpg', cv2.flip(image, 1))
      cv2.imwrite('Face.jpg', cv2.flip(img_resize, 1))
      exit(0)

    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()