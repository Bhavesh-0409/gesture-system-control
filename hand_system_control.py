import cv2
import mediapipe as mp
import time
from collections import deque
from pycaw.pycaw import AudioUtilities
import screen_brightness_control as sbc
import threading
import queue
import numpy as np


action_queue = queue.Queue()


system_state = {
    "volume": 0.5,     
    "brightness": 50,  
    "muted": False
}

def worker():
    while True:
        try:
            func, args, state_key, state_val = action_queue.get()
            if func is None: break
            func(*args)
            if state_key:
                system_state[state_key] = state_val
            action_queue.task_done()
        except Exception as e:
            print(f"Error in worker: {e}")

t = threading.Thread(target=worker, daemon=True)
t.start()

def run_async(func, args, state_key=None, state_val=None):
    action_queue.put((func, args, state_key, state_val))


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FPS, 60)


try:
    volume = AudioUtilities.GetSpeakers().EndpointVolume

    vol_levels = {
        "low": 0.2,  
        "mid": 0.5,  
        "high": 1.0  
    }
except:
    volume = None
    vol_levels = {"low": 0.2, "mid": 0.5, "high": 1.0}


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
finger_tips = [8, 12, 16, 20]


finger_history = deque(maxlen=5)
COOLDOWN = 0.4 
last_action_time = 0
current_gesture_state = -1 

window_name = "Gesture System Control - Simple"
cv2.namedWindow(window_name)


def count_fingers(lm, handedness_label):
    count = 0

    for tip in finger_tips:
        if lm[tip].y < lm[tip - 2].y:
            count += 1
   
    thumb_tip_x = lm[4].x
    thumb_ip_x = lm[3].x
    if handedness_label == "Right":
        if thumb_tip_x < thumb_ip_x: count += 1
    else: 
        if thumb_tip_x > thumb_ip_x: count += 1
    return count

def draw_ui(img, action_text, fingers):
    h, w, _ = img.shape
    cv2.rectangle(img, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.putText(img, f"Fingers: {fingers}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(img, f"Action: {action_text}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    vol_height = int(system_state["volume"] * 200)
    cv2.rectangle(img, (20, 150), (50, 350), (50, 50, 50), -1)
    cv2.rectangle(img, (20, 350 - vol_height), (50, 350), (0, 255, 0), -1)
    cv2.putText(img, "VOL", (15, 370), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    bright_height = int((system_state["brightness"] / 100) * 200)
    cv2.rectangle(img, (w - 50, 150), (w - 20, 350), (50, 50, 50), -1)
    cv2.rectangle(img, (w - 50, 350 - bright_height), (w - 20, 350), (255, 255, 0), -1)
    cv2.putText(img, "BRI", (w - 55, 370), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    if system_state["muted"]:
        cv2.putText(img, "MUTED", (w//2 - 40, 370), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

def set_vol_mute(val): 
    if volume: volume.SetMute(val, None)

def set_vol_level(val): 
    if volume: volume.SetMasterVolumeLevelScalar(val, None)

def set_bright(val): 
    sbc.set_brightness(val)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    frame.flags.writeable = False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    frame.flags.writeable = True

    action_text = "Scanning..."
    fingers = 0
    
    if result.multi_hand_landmarks:
        hand_label = result.multi_handedness[0].classification[0].label
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        fingers = count_fingers(lm, hand_label)
        finger_history.append(fingers)
        stable_fingers = max(set(finger_history), key=finger_history.count)

        now = time.time()
        if now - last_action_time > COOLDOWN:
            

            if stable_fingers == 0:
                if current_gesture_state != 0:
                    run_async(set_vol_mute, (1,), "muted", True)
                    current_gesture_state = 0
                    last_action_time = now
                action_text = "Mute"


            elif stable_fingers == 1:
                if current_gesture_state != 1:
                    run_async(set_bright, (30,), "brightness", 30)
                    current_gesture_state = 1
                    last_action_time = now
                action_text = "Brightness: 30%"

            elif stable_fingers == 2:
                if current_gesture_state != 2:
                    run_async(set_bright, (80,), "brightness", 80)
                    current_gesture_state = 2
                    last_action_time = now
                action_text = "Brightness: 80%"


            elif stable_fingers == 3:
                if current_gesture_state != 3:
                    run_async(set_vol_mute, (0,), "muted", False)
                    run_async(set_vol_level, (vol_levels["low"],), "volume", 0.2)
                    current_gesture_state = 3
                    last_action_time = now
                action_text = "Volume: 20%"


            elif stable_fingers == 4:
                if current_gesture_state != 4:
                    run_async(set_vol_mute, (0,), "muted", False)
                    run_async(set_vol_level, (vol_levels["mid"],), "volume", 0.5)
                    current_gesture_state = 4
                    last_action_time = now
                action_text = "Volume: 50%"


            elif stable_fingers == 5:
                if current_gesture_state != 5:
                    run_async(set_vol_mute, (0,), "muted", False)
                    run_async(set_vol_level, (vol_levels["high"],), "volume", 1.0)
                    current_gesture_state = 5
                    last_action_time = now
                action_text = "Volume: 100%"

    draw_ui(frame, action_text, fingers)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == 27: break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

cap.release()
cv2.destroyAllWindows()