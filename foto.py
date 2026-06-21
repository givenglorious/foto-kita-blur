import cv2
import mediapipe as mp

print(mp.__version__)
print(mp.__file__)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def is_peace_sign(hand_landmarks):
    landmarks = hand_landmarks.landmark

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    index_pip = landmarks[6]
    middle_pip = landmarks[10]
    ring_pip = landmarks[14]
    pinky_pip = landmarks[18]
    
    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_down = ring_tip.y > ring_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y

    return index_up and middle_up and ring_down and pinky_down

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    blur_mode = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            if is_peace_sign(hand_landmarks):
                blur_mode = True


    if blur_mode:
        frame = cv2.GaussianBlur(frame, (51, 51), 0)

        cv2.putText(
            frame,
            "",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    cv2.imshow("", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()