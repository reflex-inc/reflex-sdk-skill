# Observation schema for `/v1/infer` (pi0.5)

Wire format the Reflex inference endpoint accepts. Built to match the ALOHA-trained pi0.5 input shape.

## Top-level request body

```json
{
  "observation": {
    "prompt":  "transfer the cube to the target",
    "state":   [...14 floats...],
    "images":  { ...3 cameras... }
  },
  "request_id": "optional-correlation-id"
}
```

Send as `POST` with `Content-Type: application/json` and `Authorization: Bearer rfx_<key>`.

## `prompt` (string)

Free-form natural language describing the task. pi0.5 was trained with prompts like:

- `"transfer the cube to the target location"`
- `"pick up the green block"`
- `"hand the wrench to the right arm"`

Keep it under ~80 chars and task-focused. The model isn't a chatbot; verbose context degrades performance.

## `state` (14 floats)

Robot proprioceptive state. ALOHA convention: 7 joints × 2 arms.

```
[0]  arm_left  joint 1   (waist)
[1]  arm_left  joint 2   (shoulder)
[2]  arm_left  joint 3   (elbow)
[3]  arm_left  joint 4   (forearm roll)
[4]  arm_left  joint 5   (wrist angle)
[5]  arm_left  joint 6   (wrist rotate)
[6]  arm_left  gripper   (0 = open, 1 = closed; clamped 0-1)
[7]  arm_right joint 1
[8]  arm_right joint 2
[9]  arm_right joint 3
[10] arm_right joint 4
[11] arm_right joint 5
[12] arm_right joint 6
[13] arm_right gripper
```

**Single-arm robots (e.g. SO-101)**: take your live joint readings (in radians) for the available joints, fill the remaining slots with zeros. The exact convention should match how your training data was assembled — if it mirrored the 6 joints into both arms, do that; if it zero-padded, do that.

Joint values: radians for angles, normalized 0-1 for the gripper. Stay close to the demonstration range; clip to ±π if your raw encoder readings can run away.

## `images` (3 RGB cameras)

```json
{
  "cam_high":        { "encoding": "jpeg_base64", "data": "<base64>" },
  "cam_left_wrist":  { "encoding": "jpeg_base64", "data": "<base64>" },
  "cam_right_wrist": { "encoding": "jpeg_base64", "data": "<base64>" }
}
```

- **All 3 keys required**. The model expects all three even if your robot has fewer cameras; in that case duplicate the available frame into the missing slots.
- **Encoding**: `jpeg_base64` is the safe default. `png_base64` and `raw` (uint8 HxWx3 list) are also accepted but slower / heavier on the wire.
- **Resolution**: 224×224 RGB is the trained input. Higher works (resized server-side) but adds latency.
- **Color space**: BGR or RGB both work — JPEG decode normalizes to RGB.
- **Frame timing**: image must be from within ~1/control_rate of the state read. State and image more than ~50ms apart corrupts inference.

### Building the JPEG payload (Python)

```python
import base64, io
import cv2          # OR Pillow
from PIL import Image

# From OpenCV
ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64 = base64.b64encode(buf.tobytes()).decode()

# From Pillow
img = Image.fromarray(frame_rgb_uint8, mode="RGB")
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=85)
b64 = base64.b64encode(buf.getvalue()).decode()

obs["images"]["cam_high"] = {"encoding": "jpeg_base64", "data": b64}
```

JPEG quality 80–90 is the sweet spot. Below 75 visibly degrades; above 95 just bloats wire size with no policy gain.

## Response shape

```json
{
  "ok": true,
  "request_id": "...",
  "session_id": "...",
  "action_shape":   [50, 32],
  "actions_pi":     [[...32 floats...], ...50 rows...],
  "actions_aloha":  [[...14 floats...], ...50 rows...],
  "max_minus_min":  0.6675,
  "all_zero":       false,
  "infer_ms":       845.0,
  "total_ms":       1048.95,
  "model_evals":    6,
  "model_id":       "lerobot/pi05_base",
  "num_steps":      50,
  "chunk_size":     50
}
```

### `actions_pi` vs `actions_aloha`

- `actions_pi` — raw 32-dim pi0.5 output. Use if you have a custom action decoder OR a policy with a non-ALOHA action space.
- `actions_aloha` — 14-dim ALOHA-projected actions. Use this for ALOHA-class robots (most common). Maps directly to the same indices as `state`.

For SO-101 / single-arm, take `actions_aloha[t][:6]` (first 6 left-arm joints) and discard the rest.

### Sanity gates

```python
assert result["ok"]
assert not result["all_zero"]                 # all-zero chunk = adapter mismatch
assert result["max_minus_min"] > 0.05         # essentially-zero motion = adapter not loaded
assert result["infer_ms"] < 5000              # >5s typically means cold-start hang
```

## Cold-start handling

When the inference container has been idle, the first request after wake takes 30–60s and the gateway often returns HTTP 408. Retry up to 3 times with ~8s backoff:

```python
for attempt in range(3):
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 408 and attempt < 2:
            time.sleep(8); continue
        raise
```

Subsequent calls hit a warm container and complete in <2s.

## Common observation bugs

| Symptom | Cause | Fix |
|---|---|---|
| HTTP 422 from `/v1/infer` | Missing `observation.state` or wrong dimensionality | Confirm `len(state) == 14` |
| `actions_aloha[0]` all near zero | Adapter not loaded for your task | Mark your adapter active in the dashboard |
| Actions diverge from demonstrations | State outside training range | Apply calibration offsets; clamp before sending |
| First call takes 30–60s, then HTTP 408 | Cold start | Retry with backoff; subsequent calls warm |
| Inconsistent action chunks across calls | pi0.5 is a stochastic flow-matching sampler | Expected; consume action chunks deterministically downstream if needed |
