import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

# ============================================================
# CleanHands AI - Final merged version
# Logic: from latest uploaded version
# UI: keep previous fixed clean UI
# ============================================================

# -------------------- Camera / MediaPipe --------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 820)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 固定程序窗口大小
cv2.namedWindow("CleanHands AI", cv2.WINDOW_NOR5MAL)
cv2.resizeWindow("CleanHands AI", 1100, 760)

# -------------------- Parameters --------------------
TARGET_TIME = 30.0
TIME_SCORE = 30

# 动作权重
PALM_WEIGHT = 26
BACK_WEIGHT = 26
INTERLACE_WEIGHT = 14
THUMB_WEIGHT = 4

# 四个动作进度条等比例稍微加快一点
PALM_REQUIRED_TIME = 6.44
BACK_REQUIRED_TIME = 6.44
INTERLACE_REQUIRED_TIME = 5.06
THUMB_REQUIRED_TIME = 4.60

# 启动与追踪
START_THRESHOLD = 8
HISTORY_SIZE = 28
HAND_MEMORY_SEC = 0.75
SMOOTHING_ALPHA = 0.36
PREDICTION_GAIN_XY = 0.28
PREDICTION_GAIN_Z = 0.18

# 几何阈值
CONTACT_THRESHOLD = 0.155
THUMB_TOUCH_THRESHOLD = 0.065

# 动作累计需要的连续帧数
PALM_STREAK_TO_COUNT = 2
BACK_STREAK_TO_COUNT = 2
INTERLACE_STREAK_TO_COUNT = 3
THUMB_STREAK_TO_COUNT = 3

# 判定门槛
VALID_CONFIDENCE_GATE = 0.58
MIN_ACTION_SEPARATION = 0.026
MAX_DT_ADD = 0.055

# 各动作最低速度门槛
PALM_SPEED_GATE = 0.0082
BACK_SPEED_GATE = 0.0082
INTERLACE_SPEED_GATE = 0.0086
THUMB_SPEED_GATE = 0.0100

# -------------------- Runtime state --------------------
session_started = False
session_finished = False
start_time: Optional[float] = None
prev_time = 0.0
last_frame_time: Optional[float] = None
stable_detect_frames = 0

palm_time = 0.0
back_time = 0.0
interlace_time = 0.0
thumb_time = 0.0
valid_wash_time = 0.0

palm_streak = 0
back_streak = 0
interlace_streak = 0
thumb_streak = 0

last_action_text = "Waiting for two hands..."
last_valid_action = "None"

left_wrist_history = deque(maxlen=HISTORY_SIZE)
right_wrist_history = deque(maxlen=HISTORY_SIZE)
left_palm_center_history = deque(maxlen=HISTORY_SIZE)
right_palm_center_history = deque(maxlen=HISTORY_SIZE)
left_orientation_history = deque(maxlen=HISTORY_SIZE)
right_orientation_history = deque(maxlen=HISTORY_SIZE)
left_depth_history = deque(maxlen=HISTORY_SIZE)
right_depth_history = deque(maxlen=HISTORY_SIZE)
left_finger_spread_history = deque(maxlen=HISTORY_SIZE)
right_finger_spread_history = deque(maxlen=HISTORY_SIZE)

last_seen_hands: Dict[str, Optional[List[Tuple[float, float, float]]]] = {"Left": None, "Right": None}
last_seen_time = {"Left": 0.0, "Right": 0.0}
predicted_hands: Dict[str, Optional[List[Tuple[float, float, float]]]] = {"Left": None, "Right": None}
prev_seen_hands: Dict[str, Optional[List[Tuple[float, float, float]]]] = {"Left": None, "Right": None}


class PointObj:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


def reset_session() -> None:
    global session_started, session_finished, start_time, prev_time, last_frame_time, stable_detect_frames
    global palm_time, back_time, interlace_time, thumb_time, valid_wash_time
    global palm_streak, back_streak, interlace_streak, thumb_streak
    global last_action_text, last_valid_action
    global left_wrist_history, right_wrist_history, left_palm_center_history, right_palm_center_history
    global left_orientation_history, right_orientation_history, left_depth_history, right_depth_history
    global left_finger_spread_history, right_finger_spread_history
    global last_seen_hands, last_seen_time, predicted_hands, prev_seen_hands

    session_started = False
    session_finished = False
    start_time = None
    prev_time = 0.0
    last_frame_time = None
    stable_detect_frames = 0

    palm_time = 0.0
    back_time = 0.0
    interlace_time = 0.0
    thumb_time = 0.0
    valid_wash_time = 0.0

    palm_streak = 0
    back_streak = 0
    interlace_streak = 0
    thumb_streak = 0

    last_action_text = "Waiting for two hands..."
    last_valid_action = "None"

    left_wrist_history = deque(maxlen=HISTORY_SIZE)
    right_wrist_history = deque(maxlen=HISTORY_SIZE)
    left_palm_center_history = deque(maxlen=HISTORY_SIZE)
    right_palm_center_history = deque(maxlen=HISTORY_SIZE)
    left_orientation_history = deque(maxlen=HISTORY_SIZE)
    right_orientation_history = deque(maxlen=HISTORY_SIZE)
    left_depth_history = deque(maxlen=HISTORY_SIZE)
    right_depth_history = deque(maxlen=HISTORY_SIZE)
    left_finger_spread_history = deque(maxlen=HISTORY_SIZE)
    right_finger_spread_history = deque(maxlen=HISTORY_SIZE)

    last_seen_hands = {"Left": None, "Right": None}
    last_seen_time = {"Left": 0.0, "Right": 0.0}
    predicted_hands = {"Left": None, "Right": None}
    prev_seen_hands = {"Left": None, "Right": None}


def tuples_to_points(lm_tuples: List[Tuple[float, float, float]]) -> List[PointObj]:
    return [PointObj(x, y, z) for x, y, z in lm_tuples]


def dist(a: PointObj, b: PointObj) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def avg_point(points: List[PointObj]) -> Tuple[float, float]:
    x = sum(p.x for p in points) / len(points)
    y = sum(p.y for p in points) / len(points)
    return x, y


def smooth_landmarks(prev_lm: Optional[List[Tuple[float, float, float]]], new_lm, alpha: float = SMOOTHING_ALPHA):
    if prev_lm is None:
        return [(p.x, p.y, p.z) for p in new_lm]
    smoothed = []
    for i, p in enumerate(new_lm):
        px, py, pz = prev_lm[i]
        smoothed.append((
            alpha * p.x + (1 - alpha) * px,
            alpha * p.y + (1 - alpha) * py,
            alpha * p.z + (1 - alpha) * pz,
        ))
    return smoothed


def predict_landmarks(prev_lm: Optional[List[Tuple[float, float, float]]], curr_lm: Optional[List[Tuple[float, float, float]]]):
    if prev_lm is None or curr_lm is None:
        return curr_lm
    predicted = []
    for i in range(len(curr_lm)):
        px, py, pz = prev_lm[i]
        cx, cy, cz = curr_lm[i]
        predicted.append((
            cx + (cx - px) * PREDICTION_GAIN_XY,
            cy + (cy - py) * PREDICTION_GAIN_XY,
            cz + (cz - pz) * PREDICTION_GAIN_Z,
        ))
    return predicted


def movement_amount(history: deque) -> float:
    if len(history) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(history)):
        total += math.dist(history[i - 1], history[i])
    return total / (len(history) - 1)


def palm_orientation_value(lm: List[PointObj]) -> float:
    wrist = lm[0]
    index_mcp = lm[5]
    pinky_mcp = lm[17]
    v1x = index_mcp.x - wrist.x
    v1y = index_mcp.y - wrist.y
    v2x = pinky_mcp.x - wrist.x
    v2y = pinky_mcp.y - wrist.y
    return v1x * v2y - v1y * v2x


def get_hand_centers(lm: List[PointObj]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    palm_ids = [0, 1, 5, 9, 13, 17]
    finger_ids = [8, 12, 16, 20]
    palm_center = avg_point([lm[i] for i in palm_ids])
    finger_center = avg_point([lm[i] for i in finger_ids])
    return palm_center, finger_center


def palm_depth(lm: List[PointObj]) -> float:
    ids = [0, 5, 9, 13, 17]
    return sum(lm[i].z for i in ids) / len(ids)


def finger_spread_value(lm: List[PointObj]) -> float:
    tips = [lm[8], lm[12], lm[16], lm[20]]
    xs = [p.x for p in tips]
    ys = [p.y for p in tips]
    return (max(xs) - min(xs)) + 0.4 * (max(ys) - min(ys))


def pinch_thumb_check(lm: List[PointObj]) -> bool:
    thumb_tip = lm[4]
    targets = [lm[6], lm[10], lm[14], lm[18], lm[8], lm[12], lm[16], lm[20]]
    return min(dist(thumb_tip, p) for p in targets) < THUMB_TOUCH_THRESHOLD


def orientation_change_strength(history: deque) -> float:
    if len(history) < 4:
        return 0.0
    changes = 0
    for i in range(1, len(history)):
        if history[i - 1] * history[i] < 0:
            changes += 1
    mag = abs(history[-1] - history[0])
    return min(1.0, changes * 0.35 + mag * 3.8)


def depth_change_strength(history: deque) -> float:
    if len(history) < 4:
        return 0.0
    span = max(history) - min(history)
    return min(1.0, span / 0.05)


def spread_change_strength(history: deque) -> float:
    if len(history) < 4:
        return 0.0
    span = max(history) - min(history)
    return min(1.0, span / 0.15)


def speed_quality() -> Tuple[float, float]:
    s1 = movement_amount(left_wrist_history)
    s2 = movement_amount(right_wrist_history)
    s3 = movement_amount(left_palm_center_history)
    s4 = movement_amount(right_palm_center_history)
    speed = max(s1, s2, 0.9 * s3, 0.9 * s4)
    q = min(1.0, max(0.0, (speed - 0.0045) / 0.020))
    return speed, q


def overlap_quality(left_lm: List[PointObj], right_lm: List[PointObj]) -> Tuple[float, float]:
    left_palm, _ = get_hand_centers(left_lm)
    right_palm, _ = get_hand_centers(right_lm)
    center_dist = math.dist(left_palm, right_palm)
    q = 1.0 - min(1.0, center_dist / (CONTACT_THRESHOLD * 1.35))
    return center_dist, max(0.0, q)


def palm_rub_confidence(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
    center_dist, overlap_q = overlap_quality(left_lm, right_lm)
    _, speed_q = speed_quality()
    spread = (finger_spread_value(left_lm) + finger_spread_value(right_lm)) / 2
    spread_q = max(0.0, min(1.0, (spread - 0.04) / 0.22))
    center_motion = max(movement_amount(left_palm_center_history), movement_amount(right_palm_center_history))
    center_motion_q = min(1.0, max(0.0, (center_motion - 0.002) / 0.020))
    close_q = max(0.0, min(1.0, 1.0 - center_dist / (CONTACT_THRESHOLD * 1.35)))
    return 0.24 * overlap_q + 0.12 * speed_q + 0.10 * spread_q + 0.54 * max(center_motion_q, close_q)


def back_rub_confidence(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
    center_dist, _ = overlap_quality(left_lm, right_lm)
    _, speed_q = speed_quality()
    orient_q = max(
        orientation_change_strength(left_orientation_history),
        orientation_change_strength(right_orientation_history),
    )
    depth_q = max(
        max(0.0, min(1.0, (abs(palm_depth(left_lm) - palm_depth(right_lm)) - 0.003) / 0.045)),
        depth_change_strength(left_depth_history),
        depth_change_strength(right_depth_history),
    )
    compact_q = 1.0 - min(1.0, ((finger_spread_value(left_lm) + finger_spread_value(right_lm)) / 2 - 0.04) / 0.26)
    heavy_overlap_q = max(0.0, min(1.0, 1.0 - center_dist / (CONTACT_THRESHOLD * 1.15)))
    return 0.18 * heavy_overlap_q + 0.14 * speed_q + 0.34 * orient_q + 0.24 * depth_q + 0.10 * compact_q


def interlace_confidence(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
    left_tips = [left_lm[i] for i in [8, 12, 16, 20]]
    right_tips = [right_lm[i] for i in [8, 12, 16, 20]]
    left_mcp = [left_lm[i] for i in [5, 9, 13, 17]]
    right_mcp = [right_lm[i] for i in [5, 9, 13, 17]]

    center_dist, _ = overlap_quality(left_lm, right_lm)
    medium_overlap_q = 1.0 - min(1.0, abs(center_dist - CONTACT_THRESHOLD * 0.92) / (CONTACT_THRESHOLD * 0.95))

    spread_left = finger_spread_value(left_lm)
    spread_right = finger_spread_value(right_lm)
    spread_q = max(0.0, min(1.0, (((spread_left + spread_right) / 2) - 0.08) / 0.17))

    near_count = 0
    for tip in left_tips:
        near_count += sum(1 for p in right_mcp if dist(tip, p) < 0.132)
    for tip in right_tips:
        near_count += sum(1 for p in left_mcp if dist(tip, p) < 0.132)
    near_q = max(0.0, min(1.0, (near_count - 1) / 8.0))

    merged = []
    for p in left_tips:
        merged.append((p.x, 'L'))
    for p in right_tips:
        merged.append((p.x, 'R'))
    merged.sort(key=lambda t: t[0])
    alternations = 0
    for i in range(1, len(merged)):
        if merged[i][1] != merged[i - 1][1]:
            alternations += 1
    alt_q = max(0.0, min(1.0, alternations / 3.5))

    spread_motion_q = max(spread_change_strength(left_finger_spread_history), spread_change_strength(right_finger_spread_history))
    _, speed_q = speed_quality()

    return 0.14 * medium_overlap_q + 0.17 * spread_q + 0.33 * near_q + 0.18 * alt_q + 0.10 * speed_q + 0.08 * spread_motion_q


def thumb_confidence(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
    c = 0.0
    if pinch_thumb_check(left_lm):
        c += 0.5
    if pinch_thumb_check(right_lm):
        c += 0.5
    thumb_left_close = min(dist(left_lm[4], p) for p in [left_lm[6], left_lm[10], left_lm[14], left_lm[18], left_lm[8], left_lm[12], left_lm[16], left_lm[20]])
    thumb_right_close = min(dist(right_lm[4], p) for p in [right_lm[6], right_lm[10], right_lm[14], right_lm[18], right_lm[8], right_lm[12], right_lm[16], right_lm[20]])
    close_q = 1.0 - min(1.0, min(thumb_left_close, thumb_right_close) / 0.10)
    _, speed_q = speed_quality()
    return min(1.0, c * 0.62 + close_q * 0.24 + speed_q * 0.14)


def percent_from_time(value: float, required_time: float) -> int:
    return int(min(1.0, value / required_time) * 100)


def calculate_score(real_elapsed: float) -> int:
    time_part = min(real_elapsed / TARGET_TIME, 1.0) * TIME_SCORE
    palm_part = min(palm_time / PALM_REQUIRED_TIME, 1.0) * PALM_WEIGHT
    back_part = min(back_time / BACK_REQUIRED_TIME, 1.0) * BACK_WEIGHT
    interlace_part = min(interlace_time / INTERLACE_REQUIRED_TIME, 1.0) * INTERLACE_WEIGHT
    thumb_part = min(thumb_time / THUMB_REQUIRED_TIME, 1.0) * THUMB_WEIGHT
    total = int(time_part + palm_part + back_part + interlace_part + thumb_part)
    return max(0, min(total, 100))


def choose_single_action(conf_map: Dict[str, float], streak_map: Dict[str, int]):
    ranked = sorted(conf_map.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_conf = ranked[0]
    second_conf = ranked[1][1]

    speed, _ = speed_quality()
    speed_gates = {
        "Palm rubbing": PALM_SPEED_GATE,
        "Back of hands": BACK_SPEED_GATE,
        "Between fingers": INTERLACE_SPEED_GATE,
        "Thumb cleaning": THUMB_SPEED_GATE,
    }
    streak_gates = {
        "Palm rubbing": PALM_STREAK_TO_COUNT,
        "Back of hands": BACK_STREAK_TO_COUNT,
        "Between fingers": INTERLACE_STREAK_TO_COUNT,
        "Thumb cleaning": THUMB_STREAK_TO_COUNT,
    }

    best_valid = (
        best_conf >= VALID_CONFIDENCE_GATE
        and streak_map[best_name] >= streak_gates[best_name]
        and speed >= speed_gates[best_name]
        and (best_conf - second_conf) >= MIN_ACTION_SEPARATION
    )

    any_valid = False
    for action_name, conf in conf_map.items():
        if (
            conf >= VALID_CONFIDENCE_GATE
            and streak_map[action_name] >= streak_gates[action_name]
            and speed >= speed_gates[action_name]
        ):
            any_valid = True
            break

    return best_name, best_conf, best_valid, any_valid


# -------------------- UI helpers --------------------
def draw_panel(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)


def draw_text(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(
        img,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_metric_row(img, y, label, pct):
    color_ok = (140, 220, 160)
    color_mid = (170, 205, 255)
    color_low = (175, 175, 175)

    if pct >= 85:
        tag = "OK"
        tag_color = color_ok
    elif pct >= 45:
        tag = "MID"
        tag_color = color_mid
    else:
        tag = "LOW"
        tag_color = color_low

    draw_text(img, label, (44, y), 0.72, (245, 245, 245), 2)
    draw_text(img, f"{pct}%", (345, y), 0.72, (255, 255, 255), 2)
    draw_text(img, tag, (445, y), 0.68, tag_color, 2)


def draw_progress_bar(img, x, y, w, h, pct, fill_color=(230, 230, 230)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), -1)
    fill_w = int(w * max(0, min(100, pct)) / 100)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_color, -1)


def paste_camera_with_aspect(canvas, camera_frame, x, y, w, h):
    fh, fw = camera_frame.shape[:2]

    scale = min(w / fw, h / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)

    resized = cv2.resize(camera_frame, (new_w, new_h))

    offset_x = x + (w - new_w) // 2
    offset_y = y + (h - new_h) // 2

    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return offset_x, offset_y, new_w, new_h


with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.60,
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

        frame_dt = 0.0
        if last_frame_time is not None:
            frame_dt = max(0.0, cur_time - last_frame_time)
        last_frame_time = cur_time

        current_hands_map: Dict[str, List[Tuple[float, float, float]]] = {}
        track_status = "Tracking: searching"

        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                label = results.multi_handedness[idx].classification[0].label
                smoothed = smooth_landmarks(last_seen_hands[label], hand_landmarks.landmark)
                current_hands_map[label] = smoothed
                prev_seen_hands[label] = last_seen_hands[label]
                last_seen_hands[label] = smoothed
                predicted_hands[label] = predict_landmarks(prev_seen_hands[label], smoothed)
                last_seen_time[label] = cur_time

        hands_map: Dict[str, List[Tuple[float, float, float]]] = {}
        for label in ["Left", "Right"]:
            if current_hands_map.get(label) is not None:
                hands_map[label] = current_hands_map[label]
            elif predicted_hands[label] is not None and (cur_time - last_seen_time[label]) <= HAND_MEMORY_SEC:
                hands_map[label] = predicted_hands[label]
            elif last_seen_hands[label] is not None and (cur_time - last_seen_time[label]) <= HAND_MEMORY_SEC:
                hands_map[label] = last_seen_hands[label]

        real_elapsed = 0.0
        if session_started and start_time is not None:
            real_elapsed = cur_time - start_time

        real_both_hands_present = (
            current_hands_map.get("Left") is not None and
            current_hands_map.get("Right") is not None
        )

        usable_both_hands_present = (
            "Left" in hands_map and "Right" in hands_map
        )

        if ((not session_started and real_both_hands_present) or
            (session_started and usable_both_hands_present)) and not session_finished:

            if not session_started:
                stable_detect_frames += 1

            left_lm = tuples_to_points(hands_map["Left"])
            right_lm = tuples_to_points(hands_map["Right"])

            left_center, _ = get_hand_centers(left_lm)
            right_center, _ = get_hand_centers(right_lm)
            left_wrist_history.append((left_lm[0].x, left_lm[0].y))
            right_wrist_history.append((right_lm[0].x, right_lm[0].y))
            left_palm_center_history.append(left_center)
            right_palm_center_history.append(right_center)
            left_orientation_history.append(palm_orientation_value(left_lm))
            right_orientation_history.append(palm_orientation_value(right_lm))
            left_depth_history.append(palm_depth(left_lm))
            right_depth_history.append(palm_depth(right_lm))
            left_finger_spread_history.append(finger_spread_value(left_lm))
            right_finger_spread_history.append(finger_spread_value(right_lm))

            if real_both_hands_present:
                track_status = "Tracking: direct landmarks"
            else:
                track_status = "Tracking: predicted trajectory"

            if not session_started and stable_detect_frames >= START_THRESHOLD:
                session_started = True
                start_time = cur_time
                real_elapsed = 0.0

            if session_started:
                palm_conf = palm_rub_confidence(left_lm, right_lm)
                back_conf = back_rub_confidence(left_lm, right_lm)
                interlace_conf = interlace_confidence(left_lm, right_lm)
                thumb_conf = thumb_confidence(left_lm, right_lm)

                palm_streak = palm_streak + 1 if palm_conf >= 0.40 else max(0, palm_streak - 1)
                back_streak = back_streak + 1 if back_conf >= 0.43 else max(0, back_streak - 1)
                interlace_streak = interlace_streak + 1 if interlace_conf >= 0.46 else max(0, interlace_streak - 1)
                thumb_streak = thumb_streak + 1 if thumb_conf >= 0.58 else max(0, thumb_streak - 1)

                conf_map = {
                    "Palm rubbing": palm_conf,
                    "Back of hands": back_conf,
                    "Between fingers": interlace_conf,
                    "Thumb cleaning": thumb_conf,
                }
                streak_map = {
                    "Palm rubbing": palm_streak,
                    "Back of hands": back_streak,
                    "Between fingers": interlace_streak,
                    "Thumb cleaning": thumb_streak,
                }

                best_name, best_conf, is_best_valid, any_action_valid = choose_single_action(conf_map, streak_map)
                last_action_text = f"Detected: {best_name} ({int(best_conf * 100)}%)"

                if any_action_valid and frame_dt > 0:
                    valid_wash_time = min(real_elapsed, valid_wash_time + min(frame_dt, MAX_DT_ADD))

                if is_best_valid and frame_dt > 0:
                    dt_add = min(frame_dt, MAX_DT_ADD)
                    last_valid_action = best_name
                    if best_name == "Palm rubbing":
                        palm_time = min(PALM_REQUIRED_TIME, palm_time + dt_add)
                    elif best_name == "Back of hands":
                        back_time = min(BACK_REQUIRED_TIME, back_time + dt_add)
                    elif best_name == "Between fingers":
                        interlace_time = min(INTERLACE_REQUIRED_TIME, interlace_time + dt_add)
                    elif best_name == "Thumb cleaning":
                        thumb_time = min(THUMB_REQUIRED_TIME, thumb_time + dt_add)

                real_elapsed = cur_time - start_time
                if real_elapsed >= TARGET_TIME:
                    session_finished = True
        else:
            if not session_started:
                stable_detect_frames = 0

            palm_streak = max(0, palm_streak - 1)
            back_streak = max(0, back_streak - 1)
            interlace_streak = max(0, interlace_streak - 1)
            thumb_streak = max(0, thumb_streak - 1)

            if session_started and start_time is not None:
                real_elapsed = cur_time - start_time
                if real_elapsed >= TARGET_TIME:
                    session_finished = True

        if session_started and start_time is not None:
            real_elapsed = min(TARGET_TIME, cur_time - start_time)

        total_score = calculate_score(real_elapsed)
        palm_pct = percent_from_time(palm_time, PALM_REQUIRED_TIME)
        back_pct = percent_from_time(back_time, BACK_REQUIRED_TIME)
        interlace_pct = percent_from_time(interlace_time, INTERLACE_REQUIRED_TIME)
        thumb_pct = percent_from_time(thumb_time, THUMB_REQUIRED_TIME)

        # -------------------- Fixed clean UI --------------------
        win_w = 1100
        win_h = 760

        canvas = np.full((win_h, win_w, 3), 16, dtype=np.uint8)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (win_w, 170), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)

        draw_panel(canvas, 18, 190, 500, 500, (24, 24, 24))

        cam_x = 520
        cam_y = 190
        cam_w = win_w - 550
        cam_h = win_h - 230

        if cam_w < 420:
            cam_w = 420
        if cam_h < 320:
            cam_h = 320

        paste_camera_with_aspect(canvas, frame, cam_x, cam_y, cam_w, cam_h)

        if session_finished:
            draw_panel(canvas, 250, win_h - 260, min(win_w - 80, 1040), win_h - 20, (20, 20, 20))

        draw_text(canvas, "CleanHands AI", (28, 42), 1.0, (255, 255, 255), 2)
        draw_text(canvas, "Hand Wash Quality Monitor", (30, 72), 0.56, (180, 180, 180), 1)

        draw_text(canvas, f"FPS {int(fps)}", (win_w - 150, 42), 0.62, (210, 210, 210), 2)
        draw_text(canvas, f"Time {int(real_elapsed)} / {int(TARGET_TIME)} sec", (28, 112), 0.68, (255, 255, 255), 2)
        draw_text(canvas, f"Valid Wash Time {valid_wash_time:.1f} sec", (320, 112), 0.68, (200, 220, 255), 2)
        draw_text(canvas, f"Score {total_score} / 100", (28, 148), 0.92, (255, 255, 255), 2)

        draw_text(canvas, last_action_text, (700, 112), 0.62, (150, 255, 180), 2)
        draw_text(canvas, f"Valid Action: {last_valid_action}", (700, 145), 0.54, (200, 200, 200), 1)
        draw_text(canvas, track_status, (700, 168), 0.5, (180, 180, 180), 1)

        draw_text(canvas, "Wash Progress", (38, 230), 0.78, (255, 255, 255), 2)

        draw_metric_row(canvas, 280, "Palm rubbing", palm_pct)
        draw_progress_bar(canvas, 40, 292, 250, 10, palm_pct, (230, 230, 230))

        draw_metric_row(canvas, 335, "Back of hands", back_pct)
        draw_progress_bar(canvas, 40, 347, 250, 10, back_pct, (230, 230, 230))

        draw_metric_row(canvas, 390, "Between fingers", interlace_pct)
        draw_progress_bar(canvas, 40, 402, 250, 10, interlace_pct, (230, 230, 230))

        draw_metric_row(canvas, 445, "Thumb cleaning", thumb_pct)
        draw_progress_bar(canvas, 40, 457, 250, 10, thumb_pct, (230, 230, 230))

        if not session_started:
            draw_text(canvas, "Show both hands at the same time to start", (38, 545), 0.625, (255, 255, 255), 2)

        if session_finished:
            panel_left = 300
            panel_top = win_h - 200

            draw_text(canvas, f"Final Score  {total_score}/100", (panel_left, panel_top), 1.0, (255, 255, 255), 3)

            summary_lines = [
                ("Palm rubbing", palm_pct),
                ("Back of hands", back_pct),
                ("Between fingers", interlace_pct),
                ("Thumb cleaning", thumb_pct),
            ]

            y_summary = panel_top + 50
            for label, pct in summary_lines:
                mark = "OK" if pct >= 85 else "NO"
                status = "sufficient" if pct >= 85 else "insufficient"

                draw_text(canvas, f"{mark}  {label}", (panel_left + 10, y_summary), 0.65, (220, 220, 220), 2)
                draw_text(canvas, status, (panel_left + 360, y_summary), 0.65, (200, 200, 200), 2)
                y_summary += 40

            draw_text(canvas, "Press R to restart", (panel_left +350, win_h -490), 1.25, (255, 255, 255), 4)

        cv2.imshow("CleanHands AI", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord('r'), ord('R')):
            reset_session()

cap.release()
cv2.destroyAllWindows()