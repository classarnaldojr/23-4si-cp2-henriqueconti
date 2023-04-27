import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    model_complexity=0
)

cap = cv2.VideoCapture('pedra-papel-tesoura.mp4')

left_hand_gesture = None
right_hand_gesture = None

player_left = 0
player_right = 0


def identify_movement_hand_left(hand_left):
    global left_hand_gesture
    left_hand_coords = [[l.x, l.y, l.z] for l in hand_left.landmark]

    if (left_hand_coords[8][0] > 0.40 and left_hand_coords[6][0] < 0.37) or \
            (left_hand_coords[8][0] > 0.46 and left_hand_coords[6][0] < 0.50) or \
            ((0.5198 < left_hand_coords[8][0] < 0.53) and left_hand_coords[6][0] < 0.56):
        if left_hand_gesture is None or not left_hand_gesture.__eq__("tesoura"):
            left_hand_gesture = "tesoura"
            return left_hand_gesture
    elif (left_hand_coords[8][0] > 0.43 and left_hand_coords[17][1] < 0.41) or \
            (left_hand_coords[0][1] > 0.52 and left_hand_coords[17][1] > 0.56) or \
            (left_hand_coords[0][1] > 0.46 and left_hand_coords[17][1] < 0.511):
        if left_hand_gesture is None or not left_hand_gesture.__eq__("papel"):
            left_hand_gesture = "papel"
            return left_hand_gesture
    else:
        if left_hand_gesture is None or not left_hand_gesture.__eq__("pedra"):
            left_hand_gesture = "pedra"
            return left_hand_gesture


def identify_movement_hand_right(hand_right):
    global right_hand_gesture
    right_hand_coords = [[l.x, l.y, l.z] for l in hand_right.landmark]

    if (right_hand_coords[8][0] > 0.39942 and right_hand_coords[6][0] < 0.37) or \
            (right_hand_coords[8][0] > 0.46 and right_hand_coords[6][0] < 0.50) or \
            ((0.52001 < right_hand_coords[8][0] < 0.53) and right_hand_coords[6][0] < 0.56):
        if right_hand_gesture is None or not right_hand_gesture.__eq__("tesoura"):
            right_hand_gesture = "tesoura"
    elif (right_hand_coords[8][0] > 0.33 and right_hand_coords[17][1] < 0.37) or \
            (right_hand_coords[8][0] > 0.43 and right_hand_coords[17][1] < 0.45) or \
            (right_hand_coords[8][0] > 0.30 and right_hand_coords[17][1] < 0.42):
        if right_hand_gesture is None or not right_hand_gesture.__eq__("papel"):
            right_hand_gesture = "papel"
    else:
        if right_hand_gesture is None or not right_hand_gesture.__eq__("pedra"):
            right_hand_gesture = "pedra"


def validate_players_score(left_movement):
    global player_left, player_right
    if left_movement.__eq__("tesoura"):
        if right_hand_gesture.__eq__("tesoura"):
            return
        elif right_hand_gesture.__eq__("pedra"):
            player_right += 1
        elif right_hand_gesture.__eq__("papel"):
            player_left += 1
    elif left_movement.__eq__("papel"):
        if right_hand_gesture.__eq__("papel"):
            return
        elif right_hand_gesture.__eq__("pedra"):
            player_left += 1
        elif right_hand_gesture.__eq__("tesoura"):
            player_right += 1
    elif left_movement.__eq__("pedra"):
        if right_hand_gesture.__eq__("pedra"):
            return
        elif right_hand_gesture.__eq__("papel"):
            player_right += 1
        elif right_hand_gesture.__eq__("tesoura"):
            player_left += 1


def init_capture():
    while cap.isOpened():
        ret, frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = hands.process(rgb_frame)

        if len(results.multi_hand_landmarks) == 2:
            hand_left = results.multi_hand_landmarks[0]
            hand_right = results.multi_hand_landmarks[1]

            left_movement = identify_movement_hand_left(hand_left)
            identify_movement_hand_right(hand_right)

            if left_movement:
                validate_players_score(left_movement)

            cv2.putText(frame, f'Left Player: {player_left}', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Right Player: {player_right}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('checkpoint', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

init_capture()