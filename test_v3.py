import cv2
import mediapipe as mp
import time
import math
from collections import deque

# =========================
# CleanHands AI - test_v3.py
# 依赖: pip install opencv-python mediapipe
# Esc 退出 / R 重置
# =========================

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# 摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------- 参数 ----------
TARGET_TIME = 20.0
TIME_SCORE = 30
PALM_SCORE = 20
BACK_SCORE = 20
INTERLACE_SCORE = 20
THUMB_SCORE = 10

START_THRESHOLD = 12
CONTACT_THRESHOLD = 0.12
THUMB_TOUCH_THRESHOLD = 0.07
MOVEMENT_THRESHOLD = 0.012
HISTORY_SIZE = 12
HAND_MEMORY_SEC = 0.35

# ---------- 状态 ----------
session_started = False
session_finished = False
start_time = None
prev_time = 0.0
stable_detect_frames = 0

palm_frames = 0
back_frames = 0
interlace_frames = 0
thumb_frames = 0

left_wrist_history = deque(maxlen=HISTORY_SIZE)
right_wrist_history = deque(maxlen=HISTORY_SIZE)
left_palm_side_history = deque(maxlen=HISTORY_SIZE)
right_palm_side_history = deque(maxlen=HISTORY_SIZE)

# 遮挡时短时记忆
last_seen_hands = {"Left": None, "Right": None}
last_seen_time = {"Left": 0.0, "Right": 0.0}


def reset_session():
    global session_started, session_finished, start_time, prev_time, stable_detect_frames
    global palm_frames, back_frames, interlace_frames, thumb_frames
    global left_wrist_history, right_wrist_history, left_palm_side_history, right_palm_side_history
    global last_seen_hands, last_seen_time

    session_started = False
    session_finished = False
    start_time = None
    prev_time = 0.0
    stable_detect_frames = 0

    palm_frames = 0
    back_frames = 0
    interlace_frames = 0
    thumb_frames = 0

    left_wrist_history = deque(maxlen=HISTORY_SIZE)
    right_wrist_history = deque(maxlen=HISTORY_SIZE)
    left_palm_side_history = deque(maxlen=HISTORY_SIZE)
    right_palm_side_history = deque(maxlen=HISTORY_SIZE)

    last_seen_hands = {"Left": None, "Right": None}
    last_seen_time = {"Left": 0.0, "Right": 0.0}


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def avg_point(points):
    x = sum(p.x for p in points) / len(points)
    y = sum(p.y for p in points) / len(points)
    return x, y


def movement_amount(history):
    if len(history) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(history)):
        x1, y1 = history[i - 1]
        x2, y2 = history[i]
        total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return total / (len(history) - 1)


def palm_orientation_sign(lm):
    wrist = lm[0]
    index_mcp = lm[5]
    pinky_mcp = lm[17]

    v1x = index_mcp.x - wrist.x
    v1y = index_mcp.y - wrist.y
    v2x = pinky_mcp.x - wrist.x
    v2y = pinky_mcp.y - wrist.y

    cross = v1x * v2y - v1y * v2x
    return 1 if cross >= 0 else -1


def get_hand_centers(lm):
    palm_ids = [0, 1, 5, 9, 13, 17]
    finger_ids = [8, 12, 16, 20]
    palm_center = avg_point([lm[i] for i in palm_ids])
    finger_center = avg_point([lm[i] for i in finger_ids])
    return palm_center, finger_center


def pinch_thumb_check(lm):
    thumb_tip = lm[4]
    targets = [lm[6], lm[10], lm[14], lm[18], lm[8], lm[12], lm[16], lm[20]]
    min_d = min(dist(thumb_tip, p) for p in targets)
    return min_d < THUMB_TOUCH_THRESHOLD


def detect_palm_rub(left_lm, right_lm, left_hist, right_hist):
    left_palm, _ = get_hand_centers(left_lm)
    right_palm, _ = get_hand_centers(right_lm)

    contact = math.dist(left_palm, right_palm) < CONTACT_THRESHOLD
    moving = movement_amount(left_hist) > MOVEMENT_THRESHOLD and movement_amount(right_hist) > MOVEMENT_THRESHOLD
    return contact and moving


def detect_back_rub(left_sign_hist, right_sign_hist, left_hist, right_hist, left_lm, right_lm):
    if len(left_sign_hist) < 4 or len(right_sign_hist) < 4:
        return False

    left_palm, _ = get_hand_centers(left_lm)
    right_palm, _ = get_hand_centers(right_lm)
    contact = math.dist(left_palm, right_palm) < CONTACT_THRESHOLD
    moving = movement_amount(left_hist) > MOVEMENT_THRESHOLD or movement_amount(right_hist) > MOVEMENT_THRESHOLD

    left_changes = sum(1 for i in range(1, len(left_sign_hist)) if left_sign_hist[i] != left_sign_hist[i - 1])
    right_changes = sum(1 for i in range(1, len(right_sign_hist)) if right_sign_hist[i] != right_sign_hist[i - 1])

    return contact and moving and (left_changes >= 1 or right_changes >= 1)


def detect_interlace(left_lm, right_lm):
    left_tips = [left_lm[i] for i in [8, 12, 16, 20]]
    right_tips = [right_lm[i] for i in [8, 12, 16, 20]]

    left_x = sorted([p.x for p in left_tips])
    right_x = sorted([p.x for p in right_tips])

    left_palm, _ = get_hand_centers(left_lm)
    right_palm, _ = get_hand_centers(right_lm)
    palm_contact = math.dist(left_palm, right_palm) < CONTACT_THRESHOLD * 1.15

    mixed = 0
    for lx in left_x:
        mixed += sum(1 for rx in right_x if abs(lx - rx) < 0.05)

    return palm_contact and mixed >= 4


def detect_thumb_clean(left_lm, right_lm):
    return pinch_thumb_check(left_lm) or pinch_thumb_check(right_lm)


def calculate_score(elapsed):
    time_part = min(elapsed / TARGET_TIME, 1.0) * TIME_SCORE
    palm_part = min(palm_frames / 25.0, 1.0) * PALM_SCORE
    back_part = min(back_frames / 20.0, 1.0) * BACK_SCORE
    interlace_part = min(interlace_frames / 18.0, 1.0) * INTERLACE_SCORE
    thumb_part = min(thumb_frames / 12.0, 1.0) * THUMB_SCORE

    total = int(time_part + palm_part + back_part + interlace_part + thumb_part)
    total = max(0, min(total, 100))

    return total, {
        "time": int(time_part),
        "palm": int(palm_part),
        "back": int(back_part),
        "interlace": int(interlace_part),
        "thumb": int(thumb_part),
    }


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.45,
    min_tracking_confidence=0.45,
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time != 0 else 0
        prev_time = cur_time

        action_text = "Waiting for two hands..."
        elapsed = 0.0

        current_hands_map = {}

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                label = results.multi_handedness[idx].classification[0].label
                current_hands_map[label] = hand_landmarks.landmark
                last_seen_hands[label] = hand_landmarks.landmark
                last_seen_time[label] = cur_time

        # 用短时记忆补偿遮挡
        hands_map = {}
        for label in ["Left", "Right"]:
            if current_hands_map.get(label) is not None:
                hands_map[label] = current_hands_map[label]
            elif last_seen_hands[label] is not None and (cur_time - last_seen_time[label]) <= HAND_MEMORY_SEC:
                hands_map[label] = last_seen_hands[label]

        if "Left" in hands_map and "Right" in hands_map:
            stable_detect_frames += 1

            left_lm = hands_map["Left"]
            right_lm = hands_map["Right"]

            left_wrist_history.append((left_lm[0].x, left_lm[0].y))
            right_wrist_history.append((right_lm[0].x, right_lm[0].y))
            left_palm_side_history.append(palm_orientation_sign(left_lm))
            right_palm_side_history.append(palm_orientation_sign(right_lm))

            if not session_started and stable_detect_frames >= START_THRESHOLD:
                session_started = True
                start_time = cur_time

            if session_started and not session_finished:
                elapsed = cur_time - start_time

                palm_ok = detect_palm_rub(left_lm, right_lm, left_wrist_history, right_wrist_history)
                back_ok = detect_back_rub(
                    left_palm_side_history,
                    right_palm_side_history,
                    left_wrist_history,
                    right_wrist_history,
                    left_lm,
                    right_lm,
                )
                interlace_ok = detect_interlace(left_lm, right_lm)
                thumb_ok = detect_thumb_clean(left_lm, right_lm)

                if palm_ok:
                    palm_frames += 1
                if back_ok:
                    back_frames += 1
                if interlace_ok:
                    interlace_frames += 1
                if thumb_ok:
                    thumb_frames += 1

                if palm_ok:
                    action_text = "Detected: Palm rubbing"
                elif back_ok:
                    action_text = "Detected: Back of hands"
                elif interlace_ok:
                    action_text = "Detected: Between fingers"
                elif thumb_ok:
                    action_text = "Detected: Thumb cleaning"
                else:
                    action_text = "Detected: Hands present"

                if elapsed >= TARGET_TIME:
                    session_finished = True
        else:
            stable_detect_frames = max(0, stable_detect_frames - 1)

        if session_started:
            if session_finished:
                elapsed = TARGET_TIME
            else:
                elapsed = cur_time - start_time

        total_score, detail = calculate_score(elapsed)

        time_done = detail["time"] >= TIME_SCORE
        palm_done = detail["palm"] >= int(PALM_SCORE * 0.7)
        back_done = detail["back"] >= int(BACK_SCORE * 0.7)
        interlace_done = detail["interlace"] >= int(INTERLACE_SCORE * 0.7)
        thumb_done = detail["thumb"] >= int(THUMB_SCORE * 0.7)

        # UI
        cv2.rectangle(frame, (0, 0), (960, 180), (20, 20, 20), -1)
        cv2.putText(frame, "CleanHands AI - Hand Wash Score", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (820, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

        cv2.putText(frame, f"Time: {int(elapsed)} / {int(TARGET_TIME)} sec", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Score: {total_score} / 100", (12, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, action_text, (12, 144),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)

        memory_text = f"Hand memory: L={'Y' if last_seen_hands['Left'] is not None else 'N'}  R={'Y' if last_seen_hands['Right'] is not None else 'N'}"
        cv2.putText(frame, memory_text, (560, 144),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 2)

        lines = [
            f"[{'OK' if time_done else '  '}] 20 sec wash",
            f"[{'OK' if palm_done else '  '}] Palm rubbing",
            f"[{'OK' if back_done else '  '}] Back of hands",
            f"[{'OK' if interlace_done else '  '}] Between fingers",
            f"[{'OK' if thumb_done else '  '}] Thumb cleaning",
        ]

        panel_x = 12
        panel_y = 230
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (panel_x, panel_y + i * 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)

        if not session_started:
            cv2.putText(frame, "Show both hands to start", (12, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        if session_finished:
            cv2.rectangle(frame, (260, 560), (720, 660), (0, 0, 0), -1)
            cv2.putText(frame, f"Final Score: {total_score}/100", (320, 605),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(frame, "Press R to restart", (355, 642),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("CleanHands AI", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord('r') or key == ord('R'):
            reset_session()

cap.release()
cv2.destroyAllWindows()

