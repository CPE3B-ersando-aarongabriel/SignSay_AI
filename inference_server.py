# SignSay Inference Server
# ========================
# Local:   uvicorn inference_server:app --host 0.0.0.0 --port 8000 --reload
# Railway: auto-detects requirements.txt, handled by Procfile — uses $PORT env var automatically
#
# First run downloads MediaPipe .task files (~16 MB total, ~60s).
# Subsequent runs skip the download.

import os
import base64
import datetime
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Auto-download MediaPipe .task files if missing
# ---------------------------------------------------------------------------
TASK_FILES = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}

for filename, url in TASK_FILES.items():
    dest = BASE / filename
    if not dest.exists():
        print(f"[startup] Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[startup]   saved to {dest}")

# ---------------------------------------------------------------------------
# MediaPipe constants — copied verbatim from live_test.py
# ---------------------------------------------------------------------------
CORE_FACE_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    46, 53, 52, 65, 55, 70, 63, 105, 66, 107,
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
    276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
]
CORE_FACE_INDICES.sort()


def extract_keypoints(hand_res, pose_res, face_res) -> np.ndarray:
    """Extract 474-dimensional keypoint vector — verbatim from live_test.py."""
    pose = np.zeros(33 * 3)
    if pose_res.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_res.pose_landmarks[0]]
        ).flatten()

    face = np.zeros(len(CORE_FACE_INDICES) * 3)
    if face_res.face_landmarks:
        face_landmarks = face_res.face_landmarks[0]
        face = np.array(
            [[face_landmarks[i].x, face_landmarks[i].y, face_landmarks[i].z]
             for i in CORE_FACE_INDICES]
        ).flatten()

    lh, rh = np.zeros(21 * 3), np.zeros(21 * 3)
    if hand_res.hand_landmarks:
        for idx, handedness in enumerate(hand_res.handedness):
            label = handedness[0].category_name
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[idx]]
            ).flatten()
            if label == "Left":
                lh = coords
            elif label == "Right":
                rh = coords

    return np.concatenate([pose, face, lh, rh])


# ---------------------------------------------------------------------------
# Global state — loaded once at startup
# ---------------------------------------------------------------------------
keras_model = None
id_to_label: dict[int, str] = {}
landmarker_hand = None
landmarker_pose = None
landmarker_face = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global keras_model, id_to_label, landmarker_hand, landmarker_pose, landmarker_face

    print("[startup] Loading Keras model ...")
    keras_model = load_model(BASE / "signsay_engine_v6.keras")
    print("[startup]   model loaded.")

    df = pd.read_csv(BASE / "labels.csv")
    id_to_label = dict(zip(df["id"], df["label"]))
    print(f"[startup]   {len(id_to_label)} labels loaded.")

    # MediaPipe landmarker options — RunningMode.IMAGE (same as live_test.py)
    base_hand = python.BaseOptions(model_asset_path=str(BASE / "hand_landmarker.task"))
    opt_hand = vision.HandLandmarkerOptions(
        base_options=base_hand,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )

    base_pose = python.BaseOptions(model_asset_path=str(BASE / "pose_landmarker.task"))
    opt_pose = vision.PoseLandmarkerOptions(
        base_options=base_pose,
        running_mode=vision.RunningMode.IMAGE,
    )

    base_face = python.BaseOptions(model_asset_path=str(BASE / "face_landmarker.task"))
    opt_face = vision.FaceLandmarkerOptions(
        base_options=base_face,
        running_mode=vision.RunningMode.IMAGE,
    )

    landmarker_hand = vision.HandLandmarker.create_from_options(opt_hand)
    landmarker_pose = vision.PoseLandmarker.create_from_options(opt_pose)
    landmarker_face = vision.FaceLandmarker.create_from_options(opt_face)
    print("[startup] MediaPipe landmarkers ready.")

    yield

    landmarker_hand.close()
    landmarker_pose.close()
    landmarker_face.close()
    print("[shutdown] Landmarkers closed.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="SignSay Inference Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    frames: list[str]   # base64 JPEG strings (data URI prefix is stripped automatically)
    expectedSign: str
    sessionId: str


class DetectHandsRequest(BaseModel):
    frame: str  # single base64 JPEG (data URI prefix stripped automatically)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/detect-hands")
async def detect_hands(req: DetectHandsRequest):
    """Lightweight hand detection — runs only MediaPipe, no Keras model.
    Used by the frontend to confirm real sign activity before starting assessment."""
    b64 = req.frame
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"hands_detected": False, "num_hands": 0}
    except Exception:
        return {"hands_detected": False, "num_hands": 0}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res_hand = landmarker_hand.detect(mp_image)

    num_hands = len(res_hand.hand_landmarks) if res_hand.hand_landmarks else 0
    return {"hands_detected": num_hands > 0, "num_hands": num_hands}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "signsay_engine_v6",
        "classes": len(id_to_label),
    }


@app.post("/predict")
async def predict(req: PredictRequest):
    raw_sequence: list[np.ndarray] = []

    for b64_str in req.frames:
        # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(b64_str)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
        except Exception:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        res_hand = landmarker_hand.detect(mp_image)
        res_pose = landmarker_pose.detect(mp_image)
        res_face = landmarker_face.detect(mp_image)

        kp = extract_keypoints(res_hand, res_pose, res_face)
        raw_sequence.append(kp)

    # No valid frames
    if not raw_sequence:
        return {
            "success": False,
            "label": "No detection",
            "confidence": 0.0,
            "class_id": -1,
            "is_match": False,
            "expectedSign": req.expectedSign,
            "sessionId": req.sessionId,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "all_predictions": [],
        }

    # Normalise to exactly 30 frames via linear interpolation
    seq = np.array(raw_sequence)   # shape: (N, 474)
    if seq.shape[0] != 30:
        scale = 30 / seq.shape[0]
        seq = zoom(seq, (scale, 1), order=1)
        seq = seq[:30]             # clamp to exactly 30 after rounding

    # Run model
    probs = keras_model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
    best_id = int(np.argmax(probs))
    best_label = id_to_label.get(best_id, "Unknown")
    best_conf = float(probs[best_id])

    is_match = best_label.strip().upper() == req.expectedSign.strip().upper()

    top10_indices = np.argsort(probs)[::-1][:10]
    all_predictions = [
        {
            "label": id_to_label.get(int(i), "Unknown"),
            "confidence": float(probs[i]),
            "class_id": int(i),
        }
        for i in top10_indices
    ]

    return {
        "success": True,
        "label": best_label,
        "confidence": best_conf,
        "class_id": best_id,
        "is_match": is_match,
        "expectedSign": req.expectedSign,
        "sessionId": req.sessionId,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "all_predictions": all_predictions,
    }
