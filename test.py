import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model = tf.keras.models.load_model('Models/hand_gesture_model.keras')
scaler = joblib.load('Models/scaler.pkl')
classes = ['Paper', 'Stone']

model_asset_path = 'Models/hand_landmarker.task'

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_asset_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    print("🚀 Unified MobileNetV3 Test Started. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(time.time() * 1000)
        
        start_time = time.perf_counter()
        
        landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        display_text = "No Hand"
        prob_text = ""
        inference_time = 0

        if landmarker_result.hand_landmarks:
            for hand_landmarks in landmarker_result.hand_landmarks:
                landmarks = []
                for lm in hand_landmarks:
                    landmarks.extend([lm.x, lm.y])

                landmarks_scaled = scaler.transform([landmarks])
                input_data = landmarks_scaled.reshape(1, 42, 1)
                
                probs = model.predict(input_data, verbose=0)[0]
                prediction = np.argmax(probs)
                
                inference_time = (time.perf_counter() - start_time) * 1000
                
                display_text = f"Gesture: {classes[prediction]}"
                prob_text = f"Paper: {probs[0]:.2f} | Stone: {probs[1]:.2f}"

                for lm in hand_landmarks:
                    ix, iy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (ix, iy), 3, (0, 255, 0), -1)

        cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, prob_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Lat: {inference_time:.1f}ms", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        cv2.imshow("Performance Test - 42pt Unified", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()