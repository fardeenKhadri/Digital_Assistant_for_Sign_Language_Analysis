# **DASA: Digital Assistant for Sign Language Analysis**  

DASA is a real-time sign language recognition system that uses **MediaPipe**, **OpenCV**, and a trained deep learning model to classify hand gestures into predefined actions.  

## **Features**  
- Recognizes **21 different sign language gestures**  
- Uses **MediaPipe Holistic** for hand landmark detection  
- Displays **real-time FPS** for performance monitoring  
- **Custom visualization** for detected gestures  

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```sh
git clone https://github.com/fardeenKhadri/Digital_Assistant_for_Sign_Language_Analysis.git
cd Digital_Assistant_for_Sign_Language_Analysis
```

### **2. Install Dependencies**  
Requires **Python 3.7+**. Install necessary libraries:  
```sh
pip install opencv-python numpy tensorflow mediapipe
```

### **3. Place the Trained Model**  
Ensure the trained model file **`dasa.keras`** is in the project directory.  

### **4. Run the Application**  
```sh
python dasa.py
```

---

## **How It Works**  
1. Captures live video from the webcam  
2. **MediaPipe Holistic** detects hand landmarks  
3. Extracted keypoints are fed into the trained **deep learning model**  
4. Recognized gestures are displayed on the screen  
5. FPS (Frames Per Second) is shown for monitoring  

---

## **Recognized Gestures**  
The model detects the following **21 hand gestures**:  
```
friend, love, more, pain, play, stand, stop, what, front, right, left, 
up, down, now, eat, drink, super, hug, me, name, hello
```

---

## **Controls**  
- **Press 'Q'** – Quit the application  

---

## **Troubleshooting**  
**1. `FileNotFoundError: dasa.keras not found`**  
   → Ensure `dasa.keras` is in the same directory as `dasa.py`.  

**2. Camera not opening / Black screen**  
   → Check if another application is using the webcam. Restart if needed.  

**3. Low FPS**  
   → Close background apps or reduce OpenCV resolution.  

---

## **Future Improvements**  
- Support for **more gestures**  
- Optimize **real-time performance**  

---

## **License**  
This project is **open-source** under the MIT License. Contributions are welcome.  

---
