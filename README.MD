# Digital Assistant for Sign Language Analysis

## Description
This project leverages **MediaPipe Holistic** and a **TensorFlow LSTM neural network** to recognize gestures in real-time from video input captured via a webcam. The application is designed for gesture recognition and can be extended to applications such as **sign language translation** and **human-computer interaction**.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python **3.8 or above**
- TensorFlow **2.4.1**
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

### Setup Environment
Run the following command to install the required dependencies:

```bash
pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib
```

---

## Usage

1. **Start the application**: Run the main Python script to initiate gesture recognition:
   ```bash
   python path/to/script/dasa.py
   ```

2. **Perform gestures**: The application will use the webcam to capture and analyze gestures.

3. **View predictions**: The detected gesture will be displayed on the video feed in real-time.

---

## Features

✅ **Real-time gesture detection** – Uses MediaPipe for detecting and tracking hand and pose landmarks.  
✅ **Gesture classification** – Trained LSTM model classifies gestures from extracted keypoints.  
✅ **Custom dataset support** – Add and train additional gestures for enhanced recognition.  
✅ **Visual feedback** – Displays detected gestures in the video feed.

---

## Custom Dataset Creation

If you want to create a custom dataset for training new gestures:

1. Open and run the `main.ipynb` notebook.
2. Modify the dataset folder path to include your custom gestures.
3. Train the model on the new dataset.
4. Save and use the updated model for real-time recognition.

---

## Files and Scripts

- **`dasa.py`** – Main script for gesture detection.  
- **`main.ipynb`** – Jupyter Notebook for dataset creation and model training.  
- **`dasa.keras`** – Pre-trained model for gesture recognition.  
- **`MP_Data/`** – Directory storing gesture keypoints for training.  

---

## Contributing

Contributions are welcome! Follow these steps:

1. **Fork** the repository.  
2. **Create a new branch**:  
   ```bash
   git checkout -b feature-branch
   ```
3. **Commit your changes**:  
   ```bash
   git commit -am "Added new feature"
   ```
4. **Push to the branch**:  
   ```bash
   git push origin feature-branch
   ```
5. **Open a Pull Request**.

---

## License

This project is open-source and available under the **MIT License**.  

---  

For any issues or feature requests, please raise an **issue** on GitHub. 🚀
