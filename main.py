import cv2, time
import mediapipe as mp

capture = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
position_y = 40

pTime = 0
cTime = 0

# if camera fails to open
if not capture.isOpened():
    print("Fail to open the Camera")

# if the camera opens successfully
while True:
    ret, frame = capture.read()

    # if frame fails to load
    if not ret:
        print("fail to load a pixel")
        break

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    cv2.rectangle(frame, (625,10), (500, 45), (255,0,0), thickness = 2)
    cv2.putText(frame, "Score: 0", (510, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    if position_y > 500:
        position_y = 40

    cv2.circle(frame, (300, position_y), 10, (0,0,255), thickness = -1)
    position_y += 15
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            index_tip = handLms.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            cv2.ellipse(frame, (cx, cy - 45), (50, 50), 0, 0, 180, (255,0,255), 2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime  

    cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()