import os
import threading
import cv2
import mediapipe as mp
import time
import platform
import subprocess
import numpy as np
import tensorflow as tf
import joblib
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from libs.airshare.sender.FileSender import NetworkFileSender
from libs.airshare.receiver.FileReceiver import NetworkFileReceiver
from Capture import take_screenshot

class UniversalHandNode:
    def __init__(self, mp_model_path='Models/hand_landmarker.task', delay=2.5):
        
        # Mediapipe setup
        base_options = python.BaseOptions(model_asset_path=mp_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.8
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Load trained model and scaler
        self.model = tf.keras.models.load_model('Models/hand_gesture_model.keras')
        self.scaler = joblib.load('Models/scaler.pkl')
        
        # Map indices to internal logic: 0 = Paper (Palm), 1 = Stone (Fist)
        self.classes = ['Paper', 'Stone']
        
        # State Logic
        self.delay = delay
        self.last_action_time = time.time() + 2.0 
        self.current_confirmed_state = None
        self.is_busy = False
        
        # Buffering
        self.state_history = []
        self.buffer_size = 10
        
        # Threading
        self.latest_landmarks = None
        self.model_prediction = None
        self.lock = threading.Lock()
        
        threading.Thread(target=self._inference_worker, daemon=True).start()

    def _inference_worker(self):
        while True:
            if self.latest_landmarks is not None and not self.is_busy:
                with self.lock:
                    coords = self.latest_landmarks
                
                # Landmark Scaling
                # Ensure coords is a 2D array for the scaler
                coords_scaled = self.scaler.transform([coords])
                
                # CNN Input Reshaping
                input_data = coords_scaled.reshape(1, 42, 1)
                
                # Prediction
                probs = self.model.predict(input_data, verbose=0)[0]
                
                with self.lock:
                    # Choose class with highest confidence
                    self.model_prediction = self.classes[np.argmax(probs)]
            
            time.sleep(0.01)

    def process_frame(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            hand_lms = result.hand_landmarks[0]
            
            current_coords = []
            for lm in hand_lms:
                current_coords.extend([lm.x, lm.y])
            
            with self.lock:
                self.latest_landmarks = current_coords
                local_pred = self.model_prediction

            if local_pred:
                self.state_history.append(local_pred)
                if len(self.state_history) > self.buffer_size:
                    self.state_history.pop(0)

                # Open/Closed hand consensus
                if len(self.state_history) == self.buffer_size and len(set(self.state_history)) == 1:
                    new_state = self.state_history[0]
                    
                    if self.current_confirmed_state is not None and new_state != self.current_confirmed_state:
                        if time.time() - self.last_action_time > self.delay:
                            
                            # OPEN (Paper) to CLOSED (Stone) -> SEND (Grab)
                            if self.current_confirmed_state == "Paper" and new_state == "Stone":
                                self.is_busy = True
                                threading.Thread(target=self.perform_send, daemon=True).start()
                            
                            # CLOSED (Stone) to OPEN (Paper) -> RECEIVE (Release)
                            elif self.current_confirmed_state == "Stone" and new_state == "Paper":
                                self.is_busy = True
                                threading.Thread(target=self.perform_receive, daemon=True).start()

                            self.last_action_time = time.time()
                    
                    self.current_confirmed_state = new_state

        return frame

    # Perform Send
    def perform_send(self):
        print(">>> GESTURE: Grabbed! Attempting to SEND (20s timeout)...")
        take_screenshot()
        
        def send_task():
            start_time = time.time()
            success = False
            file_path = os.path.abspath('files/screenshot/screenshot.png')
            
            # do a 20 second timeout
            while time.time() - start_time < 20:
                try:
                    sender = NetworkFileSender(file_path)
                    sender.start_sending()
                    print("Send Complete!")
                    success = True
                    break 
                except Exception:
                    # Hearbeat
                    remaining = int(20 - (time.time() - start_time))
                    print(f"Searching for receiver... {remaining}s left", end='\r')
                    time.sleep(1)
            
            if not success:
                print("\n SEND ERROR: No receiver found within 20 seconds.")
            
            self.is_busy = False
            self.state_history = [] 

        threading.Thread(target=send_task, daemon=True).start()

    # Perform Receive
    def perform_receive(self):
        print(">>> GESTURE: Released! Receiving...")
        try:
            receiver = NetworkFileReceiver()
            receiver.start_server()
            receiver.listen_for_requests()
            time.sleep(1.5)
            self._open_file('received_screenshot.png')
        finally:
            self.is_busy = False
            self.state_history = []

    def _open_file(self, path):
        full_path = os.path.abspath(path)
        if not os.path.exists(full_path): return
        try:
            if platform.system() == 'Windows': os.startfile(full_path)
            elif platform.system() == 'Darwin': subprocess.call(['open', full_path])
            else: subprocess.call(['xdg-open', full_path])
        except Exception as e: print(f"Error opening file: {e}")

def main():
    cap = cv2.VideoCapture(0)
    node = UniversalHandNode()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        node.process_frame(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()