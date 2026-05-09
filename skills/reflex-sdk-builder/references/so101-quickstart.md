# SO-101 quickstart — connect a real arm to Reflex inference

Goal: hook a SO-101 single-arm robot to Reflex `pi0.5` inference and run a closed-loop "type a prompt → arm moves" demo. End-to-end <30 min after parts arrive.

The full working script is at **https://github.com/reflex-inc/quickstart** (`quickstart.py`). This doc summarizes what it does and why.

## Hardware

- SO-101 arm — 6 STS3215 servos at IDs 1–6 (motor 6 is the gripper)
- USB cable to laptop (`/dev/ttyACM0` on Linux, `/dev/cu.usbmodem*` on macOS)
- 1 USB webcam (the script duplicates this single frame into all 3 camera slots)
- Any reasonable laptop — inference happens on Reflex servers; the laptop just streams obs and actuates motors

For a multi-camera rig (1 overhead + 2 wrist-mounted), capture each one independently and pass distinct JPEGs as `cam_high`, `cam_left_wrist`, `cam_right_wrist`.

## Software

```bash
pip install reflex-sdk lerobot feetech-servo-sdk deepdiff opencv-python pillow numpy
export REFLEX_API_KEY="rfx_..."          # mint at https://app.tryreflex.ai/keys
```

Linux needs `/dev/ttyACM0` access — either `sudo -E python ...` or add your user to the `dialout` group via udev.

## Motor map

```python
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
```

## Raw step ↔ radians

SO-101 servos report 0..4095 with home ≈ 2048. pi0.5 emits radians.

```python
import math
def raw_to_rad(raw: int) -> float:
    return (raw - 2048) / 4096.0 * 2.0 * math.pi
def rad_to_raw(rad: float) -> int:
    return int(round(2048 + rad / (2 * math.pi) * 4096))
```

## Calibration

Run once to capture your home pose:

```python
import time
from lerobot.motors.feetech.feetech import FeetechMotorsBus
bus = FeetechMotorsBus(port="/dev/ttyACM0", motors=MOTORS, calibration=None,
                       protocol_version=0)
bus.connect(handshake=True)
input("Move arm to home pose, press Enter...")
home = {n: bus.read("Present_Position", n, normalize=False, num_retry=2)
        for n in MOTORS}
print("home_offsets =", home)
# Persist to ~/.reflex/so101.json so the runtime can subtract them.
```

## Closed-loop control (25 Hz)

Per-tick: read joints + capture frame → call `/v1/infer` → replay first ~10 actions of the 50-step chunk with safety clip → re-observe.

```python
import base64, cv2, math, time

NAMES = list(MOTORS)
RATE_HZ = 25
PERIOD = 1.0 / RATE_HZ
STEPS_PER_CHUNK = 10
MAX_DELTA = 200       # raw steps per tick (4096 = 360° → 200 ≈ 17°)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start_pos = {n: bus.read("Present_Position", n, normalize=False, num_retry=2) for n in NAMES}
for n in NAMES:
    bus.enable_torque(motors=[n], num_retry=1)

try:
    while True:
        prompt = input("> ").strip()
        if not prompt or prompt in {"quit", "exit", "q"}:
            break

        # 1. Read current joint state and capture a frame
        cur_raw = {n: bus.read("Present_Position", n, normalize=False, num_retry=1)
                   for n in NAMES}
        state14 = [raw_to_rad(cur_raw[n]) for n in NAMES] + [0.0] * 8
        ok, frame = cap.read()
        ok2, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(jpg.tobytes()).decode()

        # 2. Call Reflex inference (see SKILL.md for the `infer` helper)
        res = infer(prompt, state14, b64)
        if not res or not res.get("ok"):
            continue
        actions = res.get("actions_aloha") or []

        # 3. Replay first N actions with a per-tick delta clip
        for i in range(min(STEPS_PER_CHUNK, len(actions))):
            a = actions[i]
            targets = {}
            for j, name in enumerate(NAMES):
                pred_raw = rad_to_raw(a[j])
                targets[name] = max(cur_raw[name] - MAX_DELTA,
                                    min(cur_raw[name] + MAX_DELTA, pred_raw))
            bus.sync_write("Goal_Position", targets, normalize=False, num_retry=0)
            time.sleep(PERIOD)

finally:
    # ALWAYS return to start position and disable torque on shutdown
    end_pos = {n: bus.read("Present_Position", n, normalize=False, num_retry=1) for n in NAMES}
    steps = max(1, int(1.5 * RATE_HZ))
    for k in range(steps + 1):
        alpha = k / steps
        tgt = {n: int(round(end_pos[n] + alpha * (start_pos[n] - end_pos[n])))
               for n in NAMES}
        bus.sync_write("Goal_Position", tgt, normalize=False, num_retry=0)
        time.sleep(PERIOD)
    for n in NAMES:
        bus.disable_torque(motors=[n], num_retry=1)
    bus.disconnect(disable_torque=False)
    cap.release()
```

## Safety rails (mandatory)

- **`MAX_DELTA` clip per joint per tick** — never let one action step exceed ~17° of motion. This contains a misbehaving model.
- **Joint-limit clamp** — clip each joint target to your arm's mechanical range before sending. pi0.5 was trained on ALOHA range-of-motion; SO-101 has different stops.
- **Always return to start + `disable_torque` in a `finally` block** — a crashed script with torque still enabled holds the arm rigid.
- **Watchdog** — halt the loop if `infer_ms > 2000` for >3 consecutive calls (cold-start hang).
- **Test the e-stop physically** before any real task. Map it to `disable_torque` in your script.

## Smoke checklist

```bash
# 1. Motors respond
python3 -c "from lerobot.motors.feetech.feetech import FeetechMotorsBus; from lerobot.motors.motors_bus import Motor, MotorNormMode; \
  bus=FeetechMotorsBus(port='/dev/ttyACM0', motors={'gripper': Motor(6,'sts3215',MotorNormMode.RANGE_0_100)}, calibration=None, protocol_version=0); \
  bus.connect(handshake=True); print(bus.read('Present_Position','gripper', normalize=False))"

# 2. Camera captures
python3 -c "import cv2; c=cv2.VideoCapture(0); ok,f=c.read(); print(ok, f.shape if ok else None)"

# 3. Inference smoke (no actuation)
SKIP_ARM=1 python3 quickstart.py     # uses synthetic black frame

# 4. Live closed loop with one prompt
python3 quickstart.py --prompt "transfer the cube to the target"
```

## Cost notes

- Inference: a 5-min control session at 25 Hz costs roughly the price of a few seconds of GPU time.
- LoRA fine-tune: a 1-epoch run on the public `aloha_sim_transfer_cube_human` (50 episodes) is a few dollars on managed B200.
- Adapter selection: by default the inference endpoint serves the active adapter for the platform — to use YOUR fine-tuned adapter for inference, mark it active in the dashboard or contact support.

## Common SO-101 failures

| Symptom | Likely cause | Fix |
|---|---|---|
| Arm jitters wildly mid-task | state14 zero-padding inconsistent with training data | Re-record demos with the same padding convention, fine-tune |
| Gripper closes/opens randomly | gripper joint outside training range | Clamp `action[6]` to your gripper's actual range |
| First infer call takes 30–60s | warm pool was scaled to zero | Retry on HTTP 408 (3× with 8s backoff); pre-warm with one infer call before the control loop |
| Actions all zero | adapter mismatch — model loaded the wrong adapter | Check `result["model_id"]`; ensure your adapter is active |
| `arm.connect` hangs | wrong serial port or permission | Try `ls /dev/ttyACM*`; add user to `dialout` group |
