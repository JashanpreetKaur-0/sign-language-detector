import cv2
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("error: cannot access webcam")
    print("check: is your webcam cponnected?")
    exit()
print("webcam opened successfully")
print("press 'q' to quit the window")

while True:
    success ,frame=cap.read()
    if not success:
        print("could not read frame from webcam")
        break
    cv2.imshow("Webcam Test-Press Q to Quit",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("closing webcam..")
        break
cap.release()
cv2.destroyAllWindows()
print("✅ Webcam closed successfully!")