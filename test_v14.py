import cv2
import mediapipe as mp
import time
import math
from collections import deque
from typing import Dict, List, Tuple, Optional

# ============================================================
# CleanHands AI - Final Competition Version
# ------------------------------------------------------------
# 目标：
# 1. 实时检测四种洗手动作
# 2. 30 秒计时，到时结束
# 3. 界面不显示概率，只显示：
#    - 当前动作
#    - 四项完成百分比
#    - 最终评分
# 4. 同一时刻只认一个动作
# ============================================================

# -------------------- Camera / MediaPipe --------------------
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# -------------------- Parameters --------------------
TARGET_TIME = 30.0
TIME_SCORE = 30

# 动作权重（更贴近现实）
PALM_WEIGHT = 28
BACK_WEIGHT = 28
INTERLACE_WEIGHT = 10
THUMB_WEIGHT = 4

# 动作目标时间（整体稍微加快版）
PALM_REQUIRED_TIME = 6.4
BACK_REQUIRED_TIME = 6.4
INTERLACE_REQUIRED_TIME = 5.0
THUMB_REQUIRED_TIME = 4.6

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

# 动作累计所需连续帧
PALM_STREAK_TO_COUNT = 2
BACK_STREAK_TO_COUNT = 2
INTERLACE_STREAK_TO_COUNT = 3
THUMB_STREAK_TO_COUNT = 3

# 内部识别门槛（仅内部使用，不对外显示）
VALID_CONFIDENCE_GATE = 0.58
MIN_ACTION_SEPARATION = 0.026
MAX_DT_ADD = 0.055

# 最低运动门槛，防止小抖动刷分
PALM_SPEED_GATE = 0.0076
BACK_SPEED_GATE = 0.0076
INTERLACE_SPEED_GATE = 0.0086
THUMB_SPEED_GATE = 0.0100

# -------------------- Runtime State --------------------
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

current_action_text = "Waiting for two hands..."
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

last_seen_hands: Dict[str, Optional[List[Tuple[float, float, float]]]] = {
    "Left": None,
    "Right": None,
}
last_seen_time = {"Left": 0.0, "Right": 0.0}
predicted_hands: Dict[str, Optional[List[Tuple[float, float, float]]]] = {
    "Left": None,
    "Right": None,
}
prev_seen_hands: Dict[str, Optional[List[Tuple[float, float, float]]]] = {
    "Left": None,
    "Right": None,
}


class PointObj:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


def reset_session() -> None:
    global session_started, session_finished, start_time, prev_time, last_frame_time, stable_detect_frames
    global palm_time, back_time, interlace_time, thumb_time, valid_wash_time
    global palm_streak, back_streak, interlace_streak, thumb_streak
    global current_action_text, last_valid_action
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

    current_action_text = "Waiting for two hands..."
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


def smooth_landmarks(
    prev_lm: Optional[List[Tuple[float, float, float]]],
    new_lm,
    alpha: float = SMOOTHING_ALPHA,
):
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


def predict_landmarks(
    prev_lm: Optional[List[Tuple[float, float, float]]],
    curr_lm: Optional[List[Tuple[float, float, float]]],
):
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


# -------------------- Internal Match Scores --------------------
# 这些只是程序内部判断用，不给用户显示成概率

def palm_rub_score(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
    center_dist, overlap_q = overlap_quality(left_lm, right_lm)
    _, speed_q = speed_quality()

    spread = (finger_spread_value(left_lm) + finger_spread_value(right_lm)) / 2
    spread_q = max(0.0, min(1.0, (spread - 0.03) / 0.24))

    center_motion = max(
        movement_amount(left_palm_center_history),
        movement_amount(right_palm_center_history),
    )
    center_motion_q = min(1.0, max(0.0, (center_motion - 0.0015) / 0.022))

    close_q = max(0.0, min(1.0, 1.0 - center_dist / (CONTACT_THRESHOLD * 1.42)))

    # 更强调“接触 + 同平面摩擦移动”
    return 0.22 * overlap_q + 0.10 * speed_q + 0.08 * spread_q + 0.60 * max(center_motion_q, close_q)


def back_rub_score(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
    center_dist, _ = overlap_quality(left_lm, right_lm)
    _, speed_q = speed_quality()

    orient_q = max(
        orientation_change_strength(left_orientation_history),
        orientation_change_strength(right_orientation_history),
    )

    depth_q = max(
        max(0.0, min(1.0, (abs(palm_depth(left_lm) - palm_depth(right_lm)) - 0.002) / 0.050)),
        depth_change_strength(left_depth_history),
        depth_change_strength(right_depth_history),
    )

    compact_q = 1.0 - min(
        1.0,
        ((finger_spread_value(left_lm) + finger_spread_value(right_lm)) / 2 - 0.03) / 0.28,
    )

    heavy_overlap_q = max(0.0, min(1.0, 1.0 - center_dist / (CONTACT_THRESHOLD * 1.22)))

    # 更强调“翻面 + 深度变化 + 重叠”
    return 0.16 * heavy_overlap_q + 0.12 * speed_q + 0.38 * orient_q + 0.26 * depth_q + 0.08 * compact_q


def interlace_score(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
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
        merged.append((p.x, "L"))
    for p in right_tips:
        merged.append((p.x, "R"))
    merged.sort(key=lambda t: t[0])

    alternations = 0
    for i in range(1, len(merged)):
        if merged[i][1] != merged[i - 1][1]:
            alternations += 1
    alt_q = max(0.0, min(1.0, alternations / 3.5))

    spread_motion_q = max(
        spread_change_strength(left_finger_spread_history),
        spread_change_strength(right_finger_spread_history),
    )
    _, speed_q = speed_quality()

    return 0.14 * medium_overlap_q + 0.17 * spread_q + 0.33 * near_q + 0.18 * alt_q + 0.10 * speed_q + 0.08 * spread_motion_q


def thumb_score(left_lm: List[PointObj], right_lm: List[PointObj]) -> float:
    c = 0.0
    if pinch_thumb_check(left_lm):
        c += 0.5
    if pinch_thumb_check(right_lm):
        c += 0.5

    thumb_left_close = min(
        dist(left_lm[4], p) for p in [left_lm[6], left_lm[10], left_lm[14], left_lm[18], left_lm[8], left_lm[12], left_lm[16], left_lm[20]]
    )
    thumb_right_close = min(
        dist(right_lm[4], p) for p in [right_lm[6], right_lm[10], right_lm[14], right_lm[18], right_lm[8], right_lm[12], right_lm[16], right_lm[20]]
    )
    close_q = 1.0 - min(1.0, min(thumb_left_close, thumb_right_close) / 0.10)
    _, speed_q = speed_quality()
    return min(1.0, c * 0.62 + close_q * 0.24 + speed_q * 0.14)


# -------------------- Binary Detection Logic --------------------
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


def choose_detected_action(score_map: Dict[str, float], streak_map: Dict[str, int]):
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

    detected_candidates = []
    for action_name, score in score_map.items():
        if (
            score >= VALID_CONFIDENCE_GATE
            and streak_map[action_name] >= streak_gates[action_name]
            and speed >= speed_gates[action_name]
        ):
            detected_candidates.append((action_name, score))

    if not detected_candidates:
        return None

    detected_candidates.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = detected_candidates[0]

    if len(detected_candidates) >= 2:
        second_score = detected_candidates[1][1]
        if best_score - second_score < MIN_ACTION_SEPARATION:
            return None

    return best_name


def status_text(pct: int) -> str:
    if pct >= 85:
        return "OK"
    if pct >= 45:
        return "MID"
    return "LOW"


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

        if "Left" in hands_map and "Right" in hands_map and not session_finished:
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

            if current_hands_map.get("Left") is None or current_hands_map.get("Right") is None:
                track_status = "Tracking: predicted trajectory"
            else:
                track_status = "Tracking: direct landmarks"

            if not session_started and stable_detect_frames >= START_THRESHOLD:
                session_started = True
                start_time = cur_time
                real_elapsed = 0.0

            if session_started:
                palm_match = palm_rub_score(left_lm, right_lm)
                back_match = back_rub_score(left_lm, right_lm)
                interlace_match = interlace_score(left_lm, right_lm)
                thumb_match = thumb_score(left_lm, right_lm)

                palm_streak = palm_streak + 1 if palm_match >= 0.37 else max(0, palm_streak - 1)
                back_streak = back_streak + 1 if back_match >= 0.40 else max(0, back_streak - 1)
                interlace_streak = interlace_streak + 1 if interlace_match >= 0.46 else max(0, interlace_streak - 1)
                thumb_streak = thumb_streak + 1 if thumb_match >= 0.58 else max(0, thumb_streak - 1)

                score_map = {
                    "Palm rubbing": palm_match,
                    "Back of hands": back_match,
                    "Between fingers": interlace_match,
                    "Thumb cleaning": thumb_match,
                }

                streak_map = {
                    "Palm rubbing": palm_streak,
                    "Back of hands": back_streak,
                    "Between fingers": interlace_streak,
                    "Thumb cleaning": thumb_streak,
                }

                detected_action = choose_detected_action(score_map, streak_map)

                if detected_action is None:
                    current_action_text = "Current Action: None"
                else:
                    current_action_text = f"Current Action: {detected_action}"

                if detected_action is not None and frame_dt > 0:
                    dt_add = min(frame_dt, MAX_DT_ADD)
                    valid_wash_time = min(real_elapsed, valid_wash_time + dt_add)
                    last_valid_action = detected_action

                    if detected_action == "Palm rubbing":
                        palm_time = min(PALM_REQUIRED_TIME, palm_time + dt_add)
                    elif detected_action == "Back of hands":
                        back_time = min(BACK_REQUIRED_TIME, back_time + dt_add)
                    elif detected_action == "Between fingers":
                        interlace_time = min(INTERLACE_REQUIRED_TIME, interlace_time + dt_add)
                    elif detected_action == "Thumb cleaning":
                        thumb_time = min(THUMB_REQUIRED_TIME, thumb_time + dt_add)

                real_elapsed = cur_time - start_time
                if real_elapsed >= TARGET_TIME:
                    session_finished = True
        else:
            stable_detect_frames = max(0, stable_detect_frames - 1)
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
        valid_time_pct = percent_from_time(valid_wash_time, TARGET_TIME)
        palm_pct = percent_from_time(palm_time, PALM_REQUIRED_TIME)
        back_pct = percent_from_time(back_time, BACK_REQUIRED_TIME)
        interlace_pct = percent_from_time(interlace_time, INTERLACE_REQUIRED_TIME)
        thumb_pct = percent_from_time(thumb_time, THUMB_REQUIRED_TIME)

        # -------------------- UI --------------------
        cv2.rectangle(frame, (0, 0), (960, 172), (20, 20, 20), -1)
        cv2.putText(frame, "CleanHands AI - Hand Wash Score", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (835, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 2)
        cv2.putText(frame, f"Real Time: {int(real_elapsed)} / {int(TARGET_TIME)} sec", (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(frame, f"Valid Wash Time: {valid_wash_time:.1f} sec", (340, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (120, 255, 180), 2)
        cv2.putText(frame, f"Score: {total_score} / 100", (12, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
        cv2.putText(frame, current_action_text, (12, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        cv2.putText(frame, f"Last Valid Action: {last_valid_action}", (12, 152),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 255, 180), 2)
        cv2.putText(frame, track_status, (655, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 2)

        lines = [
            ("Palm rubbing", palm_pct),
            ("Back of hands", back_pct),
            ("Between fingers", interlace_pct),
            ("Thumb cleaning", thumb_pct),
        ]

        y0 = 214
        for i, (label, pct) in enumerate(lines):
            cv2.putText(
                frame,
                f"[{status_text(pct)}] {label}: {pct}%",
                (12, y0 + i * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2
            )

        if not session_started:
            cv2.putText(frame, "Show both hands to start", (12, 455),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        if session_finished:
            cv2.rectangle(frame, (240, 530), (750, 680), (0, 0, 0), -1)
            cv2.putText(frame, f"Final Score: {total_score}/100", (305, 575),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

            summary_lines = [
                ("Palm rubbing", palm_pct),
                ("Back of hands", back_pct),
                ("Between fingers", interlace_pct),
                ("Thumb cleaning", thumb_pct),
            ]

            y_summary = 610
            for label, pct in summary_lines:
                mark = "✔" if pct >= 85 else "✘"
                text = f"{mark} {label} {'sufficient' if pct >= 85 else 'insufficient'}"
                cv2.putText(frame, text, (270, y_summary),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
                y_summary += 28

        cv2.imshow("CleanHands AI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("r"), ord("R")):
            reset_session()

cap.release()
cv2.destroyAllWindows()