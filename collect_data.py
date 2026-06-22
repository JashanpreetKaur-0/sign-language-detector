import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

file = open('hand_data.csv', 'a', newline='')
writer = csv.writer(file)

cap = cv2.VideoCapture(0)

current_label = ""
sample_count = 0

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

            row = []
            for point in hand_landmarks.landmark:
                row.append(point.x)
                row.append(point.y)
                row.append(point.z)

            if current_label != "":
                row.append(current_label)
                writer.writerow(row)
                file.flush()
                sample_count += 1
                print(f"Saved sample {sample_count} for letter: {current_label}")

    # instructions on screen
    cv2.putText(frame,
                "A B C D E F G H I L O V W Y  |  S=stop  Q=quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame,
                f"Recording: {current_label if current_label else 'NOTHING - press a key'}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.putText(frame,
                f"Total saved: {sample_count}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1) & 0xFF

    # all 14 letters mapped to keyboard keys
    if key == ord('a'):
        current_label = "A"
    elif key == ord('b'):
        current_label = "B"
    elif key == ord('c'):
        current_label = "C"
    elif key == ord('d'):
        current_label = "D"
    elif key == ord('e'):
        current_label = "E"
    elif key == ord('f'):
        current_label = "F"
    elif key == ord('g'):
        current_label = "G"
    elif key == ord('h'):
        current_label = "H"
    elif key == ord('i'):
        current_label = "I"
    elif key == ord('l'):
        current_label = "L"
    elif key == ord('o'):
        current_label = "O"
    elif key == ord('v'):
        current_label = "V"
    elif key == ord('w'):
        current_label = "W"
    elif key == ord('y'):
        current_label = "Y"
    elif key == ord('s'):
        current_label = ""
        print("--- stopped recording ---")
    elif key == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()
print(f"Done! Total samples saved: {sample_count}")