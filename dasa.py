import cv2
import numpy as np
import os
import time
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('dasa.keras')

# Define gesture actions
actions = np.array(['friend','love','more','pain','play','stand','stop','what','front','right','left',
                    'up','down','now','eat','drink','super','hug','me','name','hello'])

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to perform Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Function to extract keypoints
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# Custom function to draw landmarks with grey & cyan colors
def draw_styled_landmarks(image, results):
    # Custom colors
    hand_line_color = (180, 180, 180)  # Grey
    hand_point_color = (0, 255, 255)   # Cyan

    # Left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=hand_line_color, thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=hand_point_color, thickness=2, circle_radius=2)
        )

    # Right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=hand_line_color, thickness=3, circle_radius=4),
            mp_drawing.DrawingSpec(color=hand_point_color, thickness=2, circle_radius=2)
        )

# Initialize camera and perform gesture recognition
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
prev_time = time.time()  # Initialize FPS timer

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks with updated colors
        draw_styled_landmarks(image, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Perform prediction if we have enough frames
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_action = actions[np.argmax(res)]
            
            if res[np.argmax(res)] > threshold:
                if len(sentence) == 0 or predicted_action != sentence[-1]:
                    sentence.append(predicted_action)

            if len(sentence) > 1:
                sentence = sentence[-1:]  # Keep only the latest prediction

        # Calculate FPS
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time))
        prev_time = curr_time

        # Display gesture prediction
        cv2.rectangle(image, (0, 0), (frame.shape[1], 70), (50, 50, 50), -1)  # Dark grey background
        if sentence:
            cv2.putText(image, sentence[0], (frame.shape[1] // 3, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)  # Cyan text

        # Display FPS
        cv2.putText(image, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # White FPS text

        # Show the frame
        cv2.imshow('DASA Running', image)

        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
