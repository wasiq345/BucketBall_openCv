import cv2, time, math, random
import mediapipe as mp

# Setup
capture = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

score = 0
pTime = 0
balls = []  
last_ball_spawn = time.time()

if not capture.isOpened():
    print("Failed to open the camera")

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to load a frame")
        break
    
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    h, w, _ = frame.shape
    
    # Spawn new ball every 2 seconds
    current_time = time.time()
    if current_time - last_ball_spawn > 2.0:
        new_ball = {
            'x': random.randint(50, w-50), 
            'y': 40, 
            'speed': random.randint(8, 15)
        }
        balls.append(new_ball)
        last_ball_spawn = current_time
    
    # Draw score box
    cv2.rectangle(frame, (500, 10), (625, 45), (255, 0, 0), thickness=2)
    cv2.putText(frame, f"Score: {score}", (510, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # Update and draw all balls
    balls_to_remove = []
    for i, ball in enumerate(balls):
        # Draw ball
        cv2.circle(frame, (ball['x'], ball['y']), 10, (0, 0, 255), thickness=-1)
        
        # Move ball down
        ball['y'] += ball['speed']
        
        # Remove ball if it goes off screen
        if ball['y'] > h:
            balls_to_remove.append(i)
    
    # Remove balls that went off screen 
    for i in reversed(balls_to_remove):
        balls.pop(i)
    
    # Hand detection
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            index_tip = handLms.landmark[8]
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)
            
            # Draw bucket at index finger
            cv2.ellipse(frame, (cx, cy - 45), (50, 50), 0, 0, 180, (255, 0, 255), 2)
            
            # Check collision with all balls
            balls_caught = []
            for i, ball in enumerate(balls):
                distance = math.hypot(cx - ball['x'], (cy - 45) - ball['y'])
                if distance < 25 and ball['y'] > cy - 70 and ball['y'] < cy - 20:
                    balls_caught.append(i)
                    score += 1
            
            # Remove caught balls 
            for i in reversed(balls_caught):
                balls.pop(i)
    
    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime + 1e-5)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    
    # Show frame
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

capture.release()
cv2.destroyAllWindows()