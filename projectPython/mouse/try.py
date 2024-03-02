import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Set up mediapipe hands
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        # Read image from camera
        success, image = cap.read()
        if not success:
            break

        # Convert image to RGB and process it with mediapipe hands
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw landmarks on image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the position of the index finger tip
                x, y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image.shape[1]), \
                       int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image.shape[0])

                # Move the mouse to the finger tip position
                pyautogui.moveTo(x, y)

        # Show image
        cv2.imshow('Virtual Mouse', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
