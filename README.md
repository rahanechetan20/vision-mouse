# Vision-Pointer: Real-Time Virtual Mouse Emulation 🖱️✋

Vision-Pointer is a highly responsive, touchless virtual mouse interface built with Python, OpenCV, and the MediaPipe Tasks Vision API. 

While existing webcam-based virtual pointers suffer from severe cursor dislocation when a user flexes their hand to click (a biomechanical conflict known in HCI as the *"Midas Touch"*), Vision-Pointer solves this geometrically. By implementing a **Two-Handed Segregation Architecture**, the system physically decouples the cursor's spatial anchor from the actuation trigger, achieving zero false-positive actuations and pixel-perfect click stability.

## ✨ Key Features
* **Midas Touch Elimination:** 0% cursor shift during click actuation.
* **Two-Handed Segregation:** The right hand acts strictly as the spatial driver, while the left hand operates an independent state machine for clicks, scrolling, and zooming.
* **Deterministic Biometric Heuristics:** Uses strict geometric math rather than probabilistic ML classifiers for gesture recognition, ensuring near-zero false positives and high-speed execution.
* **Jitter Reduction:** Implements an Exponential Moving Average (EMA) filter to counteract the high-frequency spatial noise inherent to frame-by-frame pose estimation.
* **Always-On-Top HUD (v2):** Features a persistent, compacted 320x240 Heads-Up Display overlay pinned to the screen corner for real-time visual feedback across workflows.
* **High Performance:** Maintains a real-time throughput of 25–30 FPS on standard consumer hardware (e.g., Apple M3 silicon) without dedicated external GPUs.

---

## 🏗️ System Architecture

The pipeline processes 42 3D hand landmarks simultaneously (21 per hand) and routes them into two parallel sub-systems:

### 1. The Anchor (Right Hand)
Responsible for continuous spatial manipulation and cursor movement.
* **Cursor Mode (Index Extended):** Tracks Landmark 8 (Index Tip), applies the EMA filter, and moves the OS cursor.
* **Scroll Mode (Index + Middle Extended):** Locks the cursor and translates vertical finger displacement into OS scroll commands.
* **Freeze Mode (Open Palm):** Suspends tracking entirely, allowing the user to safely "park" the cursor.

### 2. The Actuator (Left Hand)
A strict finite state machine (FSM) that dispatches commands exactly once per state transition.
* ✊ **Left Click:** Fist gesture.
* 👍 **Right Click:** Thumbs-up gesture (enforces strict knuckle-clearance heuristics).
* ✌️ **Double Click:** Peace sign.
* 🤏 **Dynamic Zoom:** Pinch gesture (Thumb + Index). Spreading fingers zooms in, pinching zooms out in real-time.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* A working webcam

### 1. Clone the repository

### 2. Install dependencies
```bash
pip install mediapipe opencv-python pyautogui numpy
```

### 3. Download the MediaPipe Model
Vision-Pointer requires the hand_landmarker.task model file.

- Download it from the MediaPipe Developer portal.
- Place the hand_landmarker.task file in the root directory of this project.

4. Run the Application
For the standard terminal version:
```bash
python virtual_mouse.py
```
For the v2 version featuring the persistent desktop overlay HUD:
```bash
python virtual_mouse2.py
```

🛑 Failsafe
This application uses PyAutoGUI to control your system's mouse. If you ever lose control of the cursor, swiftly move your physical mouse pointer to any of the four extreme corners of your screen. This will trigger PyAutoGUI's FailSafeException and halt the script.

👥 Authors
Chetan Rahane - University of Texas at Dallas
Dhairya Saxena - University of Texas at Dallas
