---
name: reflex-sdk-builder
description: Build robotics applications on top of the Reflex Labs API using the Python SDK. Use when the user wants to fine-tune pi0.5 or VLA policies, run inference for robot control, register HuggingFace LeRobot datasets, drive an SO-101 arm, or wire any Reflex-hosted model into their robot stack. Triggers on "reflex sdk", "reflex api", "tryreflex", "fine-tune pi0.5", "lora finetune", "robot inference", "VLA inference", "lerobot training", "SO-101 control", or imports of `reflex` (the Reflex Labs SDK, not the Reflex UI framework).
metadata:
  author: reflex-inc
  version: "1.0.0"
---

# Build with the Reflex SDK

Reflex Labs is a hosted fine-tuning + inference service for robotics VLA policies. Customers register a LeRobot-format HuggingFace dataset, fine-tune `pi0.5` (LoRA), then call inference at production latency from a robot control loop. This skill teaches an agent to build customer applications on top of the public SDK.

Canonical, end-to-end working example: **https://github.com/reflex-inc/quickstart**
Live docs: **https://docs.tryreflex.ai**
Dashboard: **https://app.tryreflex.ai**

## When to use this skill

- User imports `reflex` and the code calls `reflex.Client`, `reflex.training.*`, or talks to `tryreflex.ai`. (The Reflex UI framework is a different package — it exposes `reflex.App`/`reflex.State`/`reflex.Component`. Pick by what the code calls.)
- User mentions Reflex Labs, tryreflex, pi0.5 fine-tune, VLA inference, SO-101, LeRobot datasets, or ALOHA action chunks in a training/inference context.
- User pastes an `rfx_…` API key.
- User wants to drive a robot arm from typed prompts via a hosted policy.

## Install

```bash
pip install reflex-sdk
```

For SO-101 hardware control, also install `lerobot`, `feetech-servo-sdk`, `opencv-python`, `pillow`, `numpy`.

## Authentication

API keys are minted at https://app.tryreflex.ai/keys and start with `rfx_`. Resolve from env first, then a key file:

```python
import os

def resolve_api_key() -> str:
    val = os.environ.get("REFLEX_API_KEY", "").strip()
    if val:
        return val
    for path in (os.path.expanduser("~/.reflex/api_key"), "/etc/reflex/api_key"):
        if os.path.exists(path):
            return open(path).read().strip()
    return ""

api_key = resolve_api_key()
assert api_key.startswith("rfx_"), "Mint a key at https://app.tryreflex.ai/keys"
```

The org needs a non-zero balance for compute (top up at `/billing`).

## Training — submit a LoRA fine-tune

```python
import reflex, time

client = reflex.Client(api_key=api_key)

result = client.training.lora_finetune(
    hf_source_uri="lerobot/aloha_sim_transfer_cube_human",  # any public LeRobot dataset
    model_name=f"my-task-{int(time.time())}",
    base_model="pi0.5",
    epochs=1,
)
run_id = (result.get("trainingRunId")
          or result.get("training_run_id")
          or result.get("run_id"))
print("run:", run_id, "→ https://app.tryreflex.ai/training-jobs/" + run_id)
```

Poll status (status moves `queued → provisioning → running → succeeded`; full pi0.5 LoRA runs ~3–5 min on the managed GPU):

```python
while True:
    poll = client.training.get(run_id)
    run = poll.get("trainingRun") or poll.get("training_job") or {}
    status = run.get("status")
    print(status, run.get("progress"), run.get("stepsCompleted"),
          "init_loss=", run.get("modalInitialLoss"),
          "final_loss=", run.get("modalFinalLoss"))
    if status in {"succeeded", "failed", "stopped"}:
        break
    time.sleep(5)
```

When training succeeds, the adapter is saved to your account. To make YOUR adapter the one served by the inference endpoint for your key, use the dashboard or contact support — by default the platform serves a shared base adapter.

## Inference — POST observations, get action chunks

The inference URL is resolved through the SDK so customer code never hardcodes infrastructure hostnames:

```python
import json, urllib.request, time
from reflex._convex import convex_url as _sdk_url

def infer_url() -> str:
    base = _sdk_url(None).rstrip("/")
    # The HTTP action endpoint lives on a companion hostname — swap the suffix.
    if base.endswith(".cloud"):
        base = base[:-len(".cloud")] + ".site"
    return base + "/v1/infer"

INFER_URL = infer_url()

def infer(prompt: str, state14: list[float], jpeg_b64: str) -> dict:
    body = json.dumps({
        "observation": {
            "prompt": prompt,
            "state":  state14,                                # 14 floats, ALOHA shape
            "images": {                                       # all 3 cams required
                "cam_high":        {"encoding": "jpeg_base64", "data": jpeg_b64},
                "cam_left_wrist":  {"encoding": "jpeg_base64", "data": jpeg_b64},
                "cam_right_wrist": {"encoding": "jpeg_base64", "data": jpeg_b64},
            },
        }
    }).encode()
    req = urllib.request.Request(INFER_URL, data=body, method="POST", headers={
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())
```

Response shape (see `references/observation-schema.md` for full field list):

```json
{
  "ok": true,
  "actions_aloha": [[14 floats], ...50 rows...],
  "actions_pi":    [[32 floats], ...50 rows...],
  "infer_ms": 845, "total_ms": 1048,
  "model_id": "lerobot/pi05_base", "num_steps": 50, "chunk_size": 50,
  "max_minus_min": 0.66, "all_zero": false
}
```

**Cold-start retry:** when the inference container has been idle, the first call after wake takes 30–60s and may return HTTP 408. Retry up to 3 times with ~8s backoff; subsequent calls are <2s warm.

**Action chunks:** pi0.5 returns 50 future actions per call. The control loop consumes the first ~8–10 of them, then re-observes and calls again. This amortizes inference cost and gives the policy temporal context.

## SO-101 arm control — radians, raw steps, safety clip

SO-101 is 6 STS3215 servos at IDs 1–6 (no separate gripper joint — the gripper is motor 6). pi0.5 emits 14-D ALOHA state/action; SO-101 is single-arm with 6 DOF, so map to the left half and zero-pad:

```python
import math
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
NAMES = list(MOTORS)

bus = FeetechMotorsBus(port="/dev/ttyACM0", motors=MOTORS, calibration=None,
                       protocol_version=0)
bus.connect(handshake=True)

# Raw step (0..4095, home ≈ 2048) ↔ radians
def raw_to_rad(raw: int) -> float:
    return (raw - 2048) / 4096.0 * 2.0 * math.pi
def rad_to_raw(rad: float) -> int:
    return int(round(2048 + rad / (2 * math.pi) * 4096))

# Build state: 6 SO-101 joints + 8 zeros = 14
cur_raw = {n: bus.read("Present_Position", n, normalize=False, num_retry=1) for n in NAMES}
state14 = [raw_to_rad(cur_raw[n]) for n in NAMES] + [0.0] * 8

# Apply an action with per-step delta clip
MAX_DELTA = 200  # raw steps; 4096 = 360°, so 200 ≈ 17° per control tick
def apply_action(action14, cur_raw):
    targets = {}
    for j, name in enumerate(NAMES):
        pred = rad_to_raw(action14[j])
        targets[name] = max(cur_raw[name] - MAX_DELTA,
                            min(cur_raw[name] + MAX_DELTA, pred))
    bus.sync_write("Goal_Position", targets, normalize=False, num_retry=0)
```

Always: enable torque before motion, return to start position and `disable_torque` on shutdown, run the playback at ~25 Hz (`time.sleep(1/25)`), consume only the first ~10 of each 50-step chunk before re-observing.

## What NOT to do

- **Do not hardcode infrastructure hostnames.** Resolve the inference URL via `reflex._convex.convex_url(None)` so the SDK picks the right deployment. Customer code should only reference Reflex servers / `tryreflex.ai`.
- **Do not skip safety clipping** on arm motion. pi0.5 was trained on ALOHA range-of-motion; SO-101 has different stops. A `MAX_DELTA` per-step cap and joint-limit clamp are mandatory before writing to motors.
- **Do not write code that POSTs directly to GPU worker URLs** or stores worker tokens. The `/v1/infer` wrapper handles auth, billing, and routing — that's the public surface.
- **Do not use mocks.** The SDK targets the live API; real inference returns in <2s warm. Faking responses defeats the point.
- **Do not include the API key in source control.** Resolve from `REFLEX_API_KEY` env or `~/.reflex/api_key`.

## Files in this skill

- `SKILL.md` — this file
- `references/observation-schema.md` — exact wire format pi0.5 expects (state, images, response)
- `references/so101-quickstart.md` — full SO-101 wiring, calibration, control loop, and cost notes
