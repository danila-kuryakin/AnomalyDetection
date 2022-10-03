import cv2
import mediapipe as mp
import numpy as np

def max_x(points):
    return max(p.x for p in points)


def max_y(points):
    return max(p.y for p in points)


def min_x(points):
    return min(p.x for p in points)


def min_y(points):
    return min(p.y for p in points)


first = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

font = cv2.FONT_HERSHEY_PLAIN
font_scale = 0.7
font_thickness = 1

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        h, w, c = image.shape
        zeros = np.zeros((h, w, c), np.uint8)

        shape = zeros.shape
        maxX = int(max_x(face_landmarks.landmark) * w)
        maxY = int(max_y(face_landmarks.landmark) * h)

        minX_normalized = min_x(face_landmarks.landmark)
        minY_normalized = min_y(face_landmarks.landmark)

        minX = int(minX_normalized * w)
        minY = int(minY_normalized * h)

        maxX -= minX
        maxY -= minY
        minX = 0
        minY = 0

        coefX = w/maxX - 0.2
        coefY = h/maxY - 0.18

        maxX = w
        maxY = h

        up_right = [minX, minY]
        up_left = [maxX, minY]
        down_left = [maxX, maxY]
        down_right = [minX, maxY]

        # cv2.circle(zeros, up_right, 2, (0, 0, 255), -1)
        # cv2.circle(zeros, up_left, 2, (0, 255, 0), -1)
        # cv2.circle(zeros, down_left, 2, (0, 255, 255), -1)
        # cv2.circle(zeros, down_right, 2, (255, 0, 0), -1)

        # if(first):
        #     print(face_landmarks.landmark)

        for landmark in face_landmarks.landmark:
            landmark.x = (landmark.x - minX_normalized) * coefX + 0.025
            landmark.y = (landmark.y - minY_normalized) * coefY + 0.025

        mp_drawing.draw_landmarks(
            image=zeros,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())

        mp_drawing.draw_landmarks(
            image=zeros,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())

        # draw point numbers
        zrF = cv2.flip(zeros, 1)
        for i, (lm) in enumerate(face_landmarks.landmark):
            cv2.putText(zrF,
                        str(i),
                        (int(w - lm.x*w), int(lm.y*h)),
                        font,
                        font_scale,
                        (0, 0, 255),
                        font_thickness)
            # if i > 30:
            #     break


    zrR = cv2.resize(zrF, (900, 830))
    cv2.imshow('Mask', zrR)

    imF = cv2.flip(image, 1)
    h, w, _ = imF.shape
    imR = cv2.resize(imF, (w*4//3, h*4//3))
    cv2.imshow('Camera', imR)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

