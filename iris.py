import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def is_focused(landmarks):
    left_iris_center = landmarks[468]  # Left iris center
    right_iris_center = landmarks[473]  # Right iris center

    iris_center_avg = [(left_iris_center.x + right_iris_center.x) / 2, 
                       (left_iris_center.y + right_iris_center.y) / 2]
    
    nose_tip = landmarks[1]  # Nose tip position
    nose_to_left_iris = distance((nose_tip.x, nose_tip.y), (left_iris_center.x, left_iris_center.y))
    nose_to_right_iris = distance((nose_tip.x, nose_tip.y), (right_iris_center.x, right_iris_center.y))

    # If the difference between left and right iris distances to the nose is significant, the user is unfocused
    if abs(nose_to_left_iris - nose_to_right_iris) > 0.02:
        return False  # Unfocused
    return True  # Focused

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #mediapipe will work only in  rgb

    result = face_mesh.process(rgb_frame) #detect landmark

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            if is_focused(landmarks):
                cv2.putText(frame, "Focused", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Unfocused", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                print("unfocussed")
            
            for idx in [468, 473]: 
                x = int(landmarks[idx].x * frame.shape[1])
                y = int(landmarks[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

    cv2.imshow('Iris Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

