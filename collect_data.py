import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()

prev_x, prev_y = 0, 0

frame_timestamp = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect_for_video(mp_image, frame_timestamp)
    frame_timestamp += 1

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Index finger tip
            index_finger = hand_landmarks[8]
            x = int(index_finger.x * frame.shape[1])
            y = int(index_finger.y * frame.shape[0])

            # Smooth movement
            curr_x = prev_x + (x - prev_x) / 8
            curr_y = prev_y + (y - prev_y) / 8

            screen_x = screen_w * curr_x / frame.shape[1]
            screen_y = screen_h * curr_y / frame.shape[0]

            pyautogui.moveTo(screen_x, screen_y)

            # Thumb tip
            thumb = hand_landmarks[4]

            # Distance for click
            distance = np.sqrt(
                (index_finger.x - thumb.x) ** 2 +
                (index_finger.y - thumb.y) ** 2
            )

            if distance < 0.05:
                pyautogui.click()

            prev_x, prev_y = curr_x, curr_y

    cv2.imshow("Virtual Mouse (New)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()