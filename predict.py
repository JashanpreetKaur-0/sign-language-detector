import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import Counter

# load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# this list stores the last 15 predictions
# instead of showing each frame's guess, we show
# only what the model agrees on most across 15 frames
recent_predictions = []

stable_letter = ""   # the final stable letter shown on screen

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # collect 63 numbers
            row = []
            for point in hand_landmarks.landmark:
                row.append(point.x)
                row.append(point.y)
                row.append(point.z)

            row = np.array(row).reshape(1, -1)
            prediction = model.predict(row)[0]

            # add this frame's prediction to our recent list
            recent_predictions.append(prediction)

            # only keep the last 15 predictions
            if len(recent_predictions) > 15:
                recent_predictions.pop(0)

            # find which letter appears most in last 15 frames
            most_common = Counter(recent_predictions).most_common(1)[0]
            letter = most_common[0]      # the letter
            count = most_common[1]       # how many times it appeared

            # only show it if it appeared at least 10 out of 15 times
            # this means the model is confident, not just guessing
            if count >= 10:
                stable_letter = letter
            else:
                stable_letter = "..."    # still thinking

    else:
        # no hand detected — clear the predictions
        recent_predictions = []
        stable_letter = ""

    # show the stable letter big on screen
    cv2.putText(frame,
                f"Sign: {stable_letter if stable_letter else 'Show your hand'}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 255, 0),
                3)

    # show a small hint of what it's currently seeing
    cv2.putText(frame,
                f"Detecting...",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()