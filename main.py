import cv2
import mediapipe as mp
import time
import pyautogui as pg

cap = cv2.VideoCapture(0)
pg.FAILSAFE = False
pg.PAUSE = 0
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.6,
                      min_tracking_confidence=0.4)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    #img = cv2.imread('images/teste.png')
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    

    if results.multi_hand_landmarks:
        # draw hands
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        results.multi_hand_landmarks
        # First hand, point 8, end of index finger
        lmk = results.multi_hand_landmarks[0].landmark[8]
        
        # This will make your hand move from 100x100 (top-left) to 900x700 (bottom-right).
        # 800x600 is the game resolution, so you can move your mouse over the game screen.
        screenX = 100+ lmk.x*800
        screenY = 100+ lmk.y*600

        pg.moveTo(100+ lmk.x*800, 100+lmk.y*600)
        
        
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)) + " FPS", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)