

## Digital Assistant for Sign Language Analysis

### Description
This project utilizes MediaPipe Holistic and a TensorFlow LSTM neural network to recognize gestures in real-time from video input captured through a webcam. The application is designed to identify various gestures, which can be extended to applications in human-computer interaction, sign language translation, and more.

### Installation

#### Prerequisites
- Python 3.8 or above
- TensorFlow 2.4.1
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

#### Setup Environment
Run the following command to install all required libraries:

```bash
pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib
```



### Usage

1. **Start the application**: Run the main Python script to initiate the gesture recognition:
   ```bash
   python path/to/script/main.py
   ```

2. **Perform gestures**: The application uses the webcam to capture video input. Perform gestures within the view of the camera.

3. **View predictions**: The predicted gesture labels are displayed on the video feed in real-time.

### Features

- **Real-time gesture detection**: Using MediaPipe for detecting and tracking hand and pose landmarks.
- **Gesture classification**: Utilizes a trained LSTM model to classify the gestures based on the extracted keypoints.
- **Visual feedback**: Displays the current detected gesture on the screen.

### Customization

- **Add more gestures**: Expand the dataset and retrain the model to recognize additional gestures.
- **Adjust model parameters**: Modify the neural network structure or training parameters for potentially better accuracy or performance.

### Scripts and Files

- **`main.py`**: The main script to run for gesture detection.


### Contributing

Contributions to this project are welcome. Please adhere to the following steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a new Pull Request.


---

