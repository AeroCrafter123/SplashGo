## Core Design Thinking

Our approach is based on splitting the system into three main layers:

- **Detection Layer**: using the Raspberry Pi camera to detect both hands and recognize handwashing actions  
- **Evaluation Layer**: defining what counts as a “valid handwashing process”  
- **Interaction Layer**: sending results to the app to display progress, reminders, and scoring  

At this stage, the most important thing is not modifying the APK, but clearly defining what counts as “valid handwashing”. Otherwise, even if the model detects actions, we would not know how to evaluate them.

---

## 1. Defining a Valid Handwashing Process

For a hackathon/demo setting, instead of using complex medical standards, we design a **process-based evaluation system** that is simple and explainable.

We define the process using six steps:

- Wet hands  
- Apply soap  
- Palm-to-palm rubbing  
- Back-of-hand cleaning  
- Finger cleaning  
- Rinse / finish  

We define a “valid” handwashing process as:

- Both hands must be detected at the same time  
- Key steps must be completed in sequence  
- Each step must last for a minimum duration  
- Total washing time must reach a threshold (e.g., 20 seconds)  

### A Practical Evaluation Standard

For implementation, we use the following criteria:

- Total duration ≥ 20 seconds  
- At least 4 core actions must be completed  
- The following actions must be included:
  - Palm rubbing  
  - Back-of-hand cleaning  
  - Finger cleaning  
- Each action must last at least 2 seconds  
- If hands leave the detection area for too long, the timer pauses  

This definition is simple, clear, and easy to explain during demos.

---

## 2. Detecting Both Hands

Since we already have hand landmark detection, the next step is to move from “points” to “actions”.

Using models like MediaPipe Hands (21 keypoints per hand), we first check two basic conditions:

### (1) Whether both hands are detected

- Both hands must be detected for multiple consecutive frames  
- Detection confidence must be above a threshold  

### (2) Whether hands are in a valid washing area

- For example, within the camera center or above the sink  
- Prevents false positives when hands move randomly outside the area  

---

## 3. From Keypoints to Actions

Instead of directly recognizing complex gestures, we use:

- Feature extraction  
- State machine logic  

### State Machine Design

- IDLE (not started)  
- HANDS_PRESENT  
- PALM_RUB  
- BACK_RUB  
- FINGER_RUB  
- FINISH  

The system evaluates motion over time instead of making decisions from a single frame.

---

## 4. Action Recognition Rules

### (1) Palm-to-Palm Rubbing

- Hands are very close  
- Palms face each other  
- Small repetitive motion  
- Duration > 2 seconds  

### (2) Back-of-Hand Cleaning

- One hand covers the back of the other  
- Rubbing motion in vertical or forward/backward direction  
- Must be done for both left and right sides  

### (3) Finger Cleaning

- Hands cross or interlace  
- Finger regions overlap  
- Repetitive rubbing motion  

### (4) Finish / Rinse

- Hands open  
- Movement becomes smaller  
- Eventually leave detection area or perform a finishing gesture  

---

## 5. Feedback Design

This part gives the system a strong “product feel”.

We provide three types of feedback:

### (1) Missing Steps

- "Please perform palm rubbing"  
- "Please clean the back of your hands"  
- "Please clean between fingers"  

### (2) Insufficient Time

- "Continue washing"  
- "Please wash for at least 20 seconds"  

### (3) Invalid Hand State

- "Please place both hands in view"  
- "Keep hands within the detection area"  

---

## 6. Recommended Program Structure

We divide the system into five modules:

### Module 1: Hand Detection
- Detect left/right hands  
- Output keypoints  
- Output confidence  

### Module 2: Feature Extraction
- Distance between hands  
- Relative palm angle  
- Finger spread  
- Motion direction and speed  
- Relative hand movement  

### Module 3: Action Recognition
- palm_rub  
- back_rub  
- finger_rub  
- unknown  

### Module 4: Process State Machine
- Track current step  
- Identify missing steps  
- Determine whether valid  

### Module 5: Output / Interaction
- Current step  
- Completed steps  
- Countdown / progress  
- Feedback messages  

---

## 7. App Integration Strategy

A key practical constraint:

If we only have an APK (without source code):

- We cannot easily modify the app  
- Integration becomes difficult  

### Best Case

If we have the app source code:

- The Raspberry Pi sends data via:
  - HTTP / WebSocket / Bluetooth / MQTT  

Example:

```json
{
  "status": "washing",
  "step": "palm_rub",
  "progress": 0.45,
  "message": "Keep rubbing your palms"
}
