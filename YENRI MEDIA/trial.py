import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Function to draw landmarks on the image
def draw_landmarks_on_image(image, detection_result):
    annotated_image = image.copy()
    if detection_result.hand_landmarks:
        for landmarks in detection_result.hand_landmarks:
            for landmark in landmarks:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                cv.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
    return annotated_image

# Open the webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Load the MediaPipe model
base_options = python.BaseOptions(model_asset_path="TEST\hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to RGB format (since OpenCV reads in BGR)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    # Convert to MediaPipe image (using numpy ndarray directly)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect hands
    detection_result = detector.detect(image)

    # Convert to NumPy array (no need to pass image width and height to mp.Image)
    image_np = image.numpy_view()

    # Draw landmarks
    annotated_image = draw_landmarks_on_image(image_np, detection_result)

    # Display result
    cv.imshow("Hand Detection", cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))

    # Check if 'q' key is pressed to quit the loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv.destroyAllWindows()
