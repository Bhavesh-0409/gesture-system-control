# gesture-system-control

Gesture Based System Control

This project uses a webcam to control system volume and screen brightness using hand gestures.
It is built using Python, OpenCV, and MediaPipe.

The idea was to go beyond a basic hand detection demo and build something that actually
interacts with the operating system in a stable and reliable way.

The system runs in real time and provides visual feedback on the screen for the detected
gesture and the current system state.

What it does

- Detects one hand using MediaPipe hand landmarks
- Counts fingers and stabilizes the result over multiple frames
- Maps specific gestures to volume and brightness controls
- Uses a background worker thread to avoid blocking the video feed
- Displays live volume and brightness indicators using OpenCV

Gesture mapping

- 0 fingers (fist): Mute system volume
- 1 finger: Set screen brightness to low (30%)
- 2 fingers: Set screen brightness to high (80%)
- 3 fingers: Set system volume to low (20%)
- 4 fingers: Set system volume to medium (50%)
- 5 fingers: Set system volume to high (100%)

How it works

Each frame from the webcam is processed using MediaPipe to extract hand landmarks.
Finger counts are smoothed using a short history window to reduce noise.
Actions such as volume and brightness changes are executed asynchronously so the UI
remains responsive.

Design choices

Early versions of the project were unstable due to gesture flickering.
Instead of adding more machine learning complexity, the focus was on engineering fixes
such as temporal smoothing, cooldown logic, and clear gesture-to-action mapping.
This made the system much more usable in practice.

Tech stack

- Python 3.9
- OpenCV
- MediaPipe Hands
- pycaw (for Windows audio control)
- screen-brightness-control

How to run

python hand_system_control.py

Notes

This project is designed for Windows systems.
It works best with good lighting and a single visible hand.
