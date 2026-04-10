from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import importlib
import json
import math
import os
import shutil
import uuid

import cv2
import pandas as pd


@dataclass(slots=True)
class VideoManifest:
    session_id: str
    clip_id: str
    source_path: str
    stored_path: str
    view: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_ms: float
    orientation: str
    created_at: str


def _project_dirs(project_root: Path) -> dict[str, Path]:
    datasets_root = project_root / "datasets" / "uploads"
    return {
        "raw": datasets_root / "raw",
        "manifests": datasets_root / "manifests",
        "processed": datasets_root / "processed",
    }


def _video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_ms = (frame_count / fps * 1000.0) if fps > 0 else 0.0
    orientation = "portrait" if height > width else "landscape"
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
        "orientation": orientation,
    }


def intake_video(project_root: Path, clip_path: Path, view: str) -> Path:
    dirs = _project_dirs(project_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    session_id = datetime.now(UTC).strftime("session_%Y%m%dT%H%M%SZ")
    clip_id = f"{view}_{uuid.uuid4().hex[:8]}"
    destination_dir = dirs["raw"] / session_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{clip_id}{clip_path.suffix.lower()}"
    shutil.copy2(clip_path, destination)

    metadata = _video_metadata(destination)
    manifest = VideoManifest(
        session_id=session_id,
        clip_id=clip_id,
        source_path=str(clip_path.resolve()),
        stored_path=str(destination.resolve()),
        view=view,
        fps=metadata["fps"],
        frame_count=metadata["frame_count"],
        width=metadata["width"],
        height=metadata["height"],
        duration_ms=metadata["duration_ms"],
        orientation=metadata["orientation"],
        created_at=datetime.now(UTC).isoformat(),
    )

    manifest_path = dirs["manifests"] / f"{session_id}_{clip_id}.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    return manifest_path


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def _processed_session_dir(project_root: Path, session_id: str) -> Path:
    session_dir = _project_dirs(project_root)["processed"] / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _load_mediapipe():
    os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".mplcache").resolve()))
    import mediapipe as mp  # type: ignore

    return mp


def _load_yolo():
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".yolo_config").resolve()))
    from ultralytics import YOLO  # type: ignore

    return YOLO


def _resolve_mediapipe_model(project_root: Path, explicit_path: str | None) -> Path | None:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    models_dir = project_root / "datasets" / "models" / "mediapipe"
    candidates.extend(
        [
            models_dir / "pose_landmarker_heavy.task",
            models_dir / "pose_landmarker_full.task",
            models_dir / "pose_landmarker_lite.task",
        ]
    )

    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (project_root / candidate)
        if resolved.exists():
            return resolved.resolve()
    return None


def _init_mediapipe_pose_runtime(project_root: Path, model_path: str | None) -> dict[str, Any] | None:
    mp = _load_mediapipe()
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        pose_model = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return {
            "backend": "mediapipe_solutions",
            "mode": "legacy",
            "mp": mp,
            "model": pose_model,
        }

    resolved_model = _resolve_mediapipe_model(project_root, model_path)
    if resolved_model is None:
        return None

    mp_python = importlib.import_module("mediapipe.tasks.python")
    vision = importlib.import_module("mediapipe.tasks.python.vision")
    base_options = mp_python.BaseOptions(model_asset_path=str(resolved_model))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    pose_model = vision.PoseLandmarker.create_from_options(options)
    return {
        "backend": "mediapipe_tasks",
        "mode": "tasks",
        "mp": mp,
        "vision": vision,
        "model": pose_model,
        "model_path": str(resolved_model),
    }


def _detect_person_bbox(frame: Any, hog: cv2.HOGDescriptor) -> tuple[int, int, int, int] | None:
    rects, weights = hog.detectMultiScale(
        frame,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )
    if len(rects) == 0:
        return None

    best_index = max(range(len(rects)), key=lambda idx: float(weights[idx]))
    x, y, w, h = rects[best_index]
    return int(x), int(y), int(w), int(h)


def _detect_ball(frame: Any) -> dict[str, float] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = (5, 110, 110)
    upper = (25, 255, 255)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 30:
        return None

    (x, y), radius = cv2.minEnclosingCircle(contour)
    circularity = 0.0
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        circularity = 4 * math.pi * area / (perimeter * perimeter)
    confidence = min(0.99, max(0.15, circularity))
    return {"x": float(x), "y": float(y), "radius": float(radius), "confidence": float(confidence)}


def _detect_ball_yolo(frame: Any, yolo_model: Any) -> dict[str, float] | None:
    results = yolo_model.predict(frame, verbose=False, imgsz=960, conf=0.15, classes=[32])
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    best_index = int(confs.argmax())
    x1, y1, x2, y2 = xyxy[best_index]
    radius = max((x2 - x1), (y2 - y1)) / 2.0
    return {
        "x": float((x1 + x2) / 2.0),
        "y": float((y1 + y2) / 2.0),
        "radius": float(radius),
        "confidence": float(confs[best_index]),
    }


def _detect_pose_yolo(frame: Any, pose_model: Any, handedness: str) -> tuple[dict[str, dict[str, float]], tuple[int, int, int, int] | None] | None:
    results = pose_model.predict(frame, verbose=False, imgsz=960, conf=0.2)
    if not results:
        return None

    result = results[0]
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None

    xy = keypoints.xy.cpu().numpy()
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.conf is not None and len(boxes.conf) == len(xy):
        confidences = boxes.conf.cpu().numpy()
        best_index = int(confidences.argmax())
    else:
        best_index = 0

    points = xy[best_index]
    coco_map = {
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }
    extracted: dict[str, dict[str, float]] = {}
    for name, index in coco_map.items():
        x, y = points[index]
        if x <= 0 and y <= 0:
            continue
        extracted[name] = {"x": float(x), "y": float(y)}

    required = {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
    if not required.issubset(extracted.keys()):
        return None

    handed = "right" if handedness.lower().startswith("r") else "left"
    extracted[f"{handed}_shooting_line"] = extracted[f"{handed}_wrist"]
    return extracted, _bbox_from_keypoints(extracted)


def _pseudo_pose_from_bbox(
    bbox: tuple[int, int, int, int],
    ball: dict[str, float] | None,
    handedness: str,
) -> dict[str, dict[str, float]]:
    x, y, w, h = bbox
    cx = x + w / 2
    shoulder_y = y + h * 0.22
    hip_y = y + h * 0.53
    knee_y = y + h * 0.77
    ankle_y = y + h * 0.98

    shoulder_half = w * 0.18
    hip_half = w * 0.14
    elbow_drop = h * 0.12
    wrist_drop = h * 0.07
    handed = "right" if handedness.lower().startswith("r") else "left"
    shoot_sign = 1 if handed == "right" else -1

    wrist_x = cx + shoot_sign * w * 0.17
    wrist_y = y + h * 0.32
    if ball:
        wrist_x = ball["x"] - shoot_sign * 12.0
        wrist_y = ball["y"] + 6.0

    elbow_x = wrist_x - shoot_sign * w * 0.08
    elbow_y = wrist_y + elbow_drop
    shoulder_x = cx + shoot_sign * shoulder_half * 0.75

    return {
        "left_shoulder": {"x": cx - shoulder_half, "y": shoulder_y},
        "right_shoulder": {"x": cx + shoulder_half, "y": shoulder_y},
        "left_hip": {"x": cx - hip_half, "y": hip_y},
        "right_hip": {"x": cx + hip_half, "y": hip_y},
        "left_knee": {"x": cx - hip_half * 0.9, "y": knee_y},
        "right_knee": {"x": cx + hip_half * 0.9, "y": knee_y},
        "left_ankle": {"x": cx - hip_half * 0.8, "y": ankle_y},
        "right_ankle": {"x": cx + hip_half * 0.8, "y": ankle_y},
        f"{handed}_shoulder": {"x": shoulder_x, "y": shoulder_y},
        f"{handed}_elbow": {"x": elbow_x, "y": elbow_y},
        f"{handed}_wrist": {"x": wrist_x, "y": wrist_y},
    }


def _bbox_from_keypoints(keypoints: dict[str, dict[str, float]]) -> tuple[int, int, int, int] | None:
    xs = [point["x"] for point in keypoints.values()]
    ys = [point["y"] for point in keypoints.values()]
    if not xs or not ys:
        return None
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return int(x_min), int(y_min), int(max(1.0, x_max - x_min)), int(max(1.0, y_max - y_min))


def _detect_pose_mediapipe(
    frame: Any,
    pose_runtime: dict[str, Any],
    handedness: str,
    timestamp_ms: int,
) -> tuple[dict[str, dict[str, float]], tuple[int, int, int, int] | None] | None:
    mp = pose_runtime["mp"]
    height, width = frame.shape[:2]

    if pose_runtime["mode"] == "legacy":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_runtime["model"].process(rgb)
        if not results.pose_landmarks:
            return None
        raw_landmarks = results.pose_landmarks.landmark

        def point(index: int) -> dict[str, float]:
            landmark = raw_landmarks[index]
            return {"x": float(landmark.x * width), "y": float(landmark.y * height)}

        pose_landmark = mp.solutions.pose.PoseLandmark
    else:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = pose_runtime["model"].detect_for_video(image, timestamp_ms)
        if not results.pose_landmarks:
            return None
        raw_landmarks = results.pose_landmarks[0]

        def point(index: int) -> dict[str, float]:
            landmark = raw_landmarks[index]
            return {"x": float(landmark.x * width), "y": float(landmark.y * height)}

        pose_landmark = pose_runtime["vision"].PoseLandmark

    keypoints = {
        "left_shoulder": point(int(pose_landmark.LEFT_SHOULDER)),
        "right_shoulder": point(int(pose_landmark.RIGHT_SHOULDER)),
        "left_elbow": point(int(pose_landmark.LEFT_ELBOW)),
        "right_elbow": point(int(pose_landmark.RIGHT_ELBOW)),
        "left_wrist": point(int(pose_landmark.LEFT_WRIST)),
        "right_wrist": point(int(pose_landmark.RIGHT_WRIST)),
        "left_hip": point(int(pose_landmark.LEFT_HIP)),
        "right_hip": point(int(pose_landmark.RIGHT_HIP)),
        "left_knee": point(int(pose_landmark.LEFT_KNEE)),
        "right_knee": point(int(pose_landmark.RIGHT_KNEE)),
        "left_ankle": point(int(pose_landmark.LEFT_ANKLE)),
        "right_ankle": point(int(pose_landmark.RIGHT_ANKLE)),
    }

    handed = "right" if handedness.lower().startswith("r") else "left"
    keypoints[f"{handed}_shooting_line"] = keypoints[f"{handed}_wrist"]
    return keypoints, _bbox_from_keypoints(keypoints)


def run_builtin_teacher(
    project_root: Path,
    manifest_path: Path,
    athlete_profile_path: Path,
    frame_stride: int = 2,
) -> dict[str, Path]:
    manifest_data = json.loads(manifest_path.read_text())
    manifest = VideoManifest(**manifest_data)
    athlete_profile = json.loads(athlete_profile_path.read_text())
    capture = cv2.VideoCapture(manifest.stored_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open stored clip: {manifest.stored_path}")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    pose_frames: list[dict[str, Any]] = []
    ball_frames: list[dict[str, Any]] = []
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % max(frame_stride, 1) != 0:
            frame_index += 1
            continue

        bbox = _detect_person_bbox(frame, hog)
        ball = _detect_ball(frame)
        if bbox:
            pose_frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": frame_index / max(manifest.fps, 1.0) * 1000.0,
                    "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
                    "keypoints": _pseudo_pose_from_bbox(bbox, ball, athlete_profile.get("handedness", "right")),
                }
            )
        if ball:
            ball_frames.append(
                {
                    "frame_index": frame_index,
                    "x": ball["x"],
                    "y": ball["y"],
                    "radius": ball["radius"],
                    "confidence": ball["confidence"],
                }
            )
        frame_index += 1

    capture.release()
    session_dir = _processed_session_dir(project_root, manifest.session_id)
    pose_json = session_dir / f"{manifest.clip_id}_teacher_pose.json"
    ball_json = session_dir / f"{manifest.clip_id}_teacher_ball.json"
    pose_json.write_text(json.dumps({"backend": "builtin_cv_teacher", "frames": pose_frames}, indent=2))
    ball_json.write_text(json.dumps({"backend": "builtin_cv_teacher", "frames": ball_frames}, indent=2))
    return {"pose_json": pose_json, "ball_json": ball_json}


def run_strong_teacher(
    project_root: Path,
    manifest_path: Path,
    athlete_profile_path: Path,
    frame_stride: int = 2,
    yolo_weights: str = "yolov8n.pt",
    pose_weights: str = "yolov8n-pose.pt",
    mediapipe_model: str | None = None,
) -> dict[str, Path]:
    manifest_data = json.loads(manifest_path.read_text())
    manifest = VideoManifest(**manifest_data)
    athlete_profile = json.loads(athlete_profile_path.read_text())
    capture = cv2.VideoCapture(manifest.stored_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open stored clip: {manifest.stored_path}")

    YOLO = _load_yolo()
    yolo_model = YOLO(yolo_weights)
    pose_runtime = _init_mediapipe_pose_runtime(project_root, mediapipe_model)
    yolo_pose_model = None
    pose_backend = "mediapipe_pose"
    pose_fallback_reason: str | None = None
    if pose_runtime is None:
        try:
            yolo_pose_model = YOLO(pose_weights)
            pose_backend = "yolov8_pose_fallback"
        except Exception as exc:
            pose_backend = "builtin_cv_pose_fallback"
            pose_fallback_reason = str(exc)
    else:
        pose_backend = pose_runtime["backend"]

    pose_frames: list[dict[str, Any]] = []
    ball_frames: list[dict[str, Any]] = []
    frame_index = 0
    handedness = athlete_profile.get("handedness", "right")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % max(frame_stride, 1) != 0:
            frame_index += 1
            continue

        timestamp_ms = int(frame_index / max(manifest.fps, 1.0) * 1000.0)
        pose_detected = None
        if pose_runtime is not None:
            pose_detected = _detect_pose_mediapipe(frame, pose_runtime, handedness, timestamp_ms)
        elif yolo_pose_model is not None:
            pose_detected = _detect_pose_yolo(frame, yolo_pose_model, handedness)
        ball = _detect_ball_yolo(frame, yolo_model) or _detect_ball(frame)
        bbox: tuple[int, int, int, int] | None = None
        keypoints: dict[str, dict[str, float]] | None = None

        if pose_detected:
            keypoints, bbox = pose_detected
        else:
            bbox = _detect_person_bbox(frame, hog)
            if bbox:
                keypoints = _pseudo_pose_from_bbox(bbox, ball, handedness)

        if keypoints:
            pose_frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": float(timestamp_ms),
                    "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]} if bbox else None,
                    "keypoints": keypoints,
                }
            )
        if ball:
            ball_frames.append(
                {
                    "frame_index": frame_index,
                    "x": ball["x"],
                    "y": ball["y"],
                    "radius": ball["radius"],
                    "confidence": ball["confidence"],
                }
            )
        frame_index += 1

    capture.release()
    if pose_runtime is not None and hasattr(pose_runtime["model"], "close"):
        pose_runtime["model"].close()
    session_dir = _processed_session_dir(project_root, manifest.session_id)
    pose_json = session_dir / f"{manifest.clip_id}_teacher_pose.json"
    ball_json = session_dir / f"{manifest.clip_id}_teacher_ball.json"
    pose_json.write_text(
        json.dumps(
            {
                "backend": pose_backend,
                "requested_backend": "mediapipe_yolov8_teacher",
                "mediapipe_model": str(_resolve_mediapipe_model(project_root, mediapipe_model)) if pose_runtime is not None and pose_runtime["mode"] == "tasks" else None,
                "pose_fallback_reason": pose_fallback_reason,
                "frames": pose_frames,
            },
            indent=2,
        )
    )
    ball_json.write_text(json.dumps({"backend": "yolov8_ball", "frames": ball_frames}, indent=2))
    return {"pose_json": pose_json, "ball_json": ball_json}


def _distance(a: dict[str, float], b: dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def _angle(a: dict[str, float], joint: dict[str, float], c: dict[str, float]) -> float:
    ab = (a["x"] - joint["x"], a["y"] - joint["y"])
    cb = (c["x"] - joint["x"], c["y"] - joint["y"])
    denom = math.sqrt(ab[0] ** 2 + ab[1] ** 2) * math.sqrt(cb[0] ** 2 + cb[1] ** 2)
    if denom == 0:
        return 180.0
    cosine = max(-1.0, min(1.0, (ab[0] * cb[0] + ab[1] * cb[1]) / denom))
    return math.degrees(math.acos(cosine))


def _opt_point(frame: dict[str, Any], key: str) -> dict[str, float] | None:
    keypoints = frame.get("keypoints", {})
    point = keypoints.get(key)
    if not point:
        return None
    return {"x": float(point["x"]), "y": float(point["y"])}


def _point(frame: dict[str, Any], key: str) -> dict[str, float]:
    point = _opt_point(frame, key)
    if point is None:
        raise KeyError(f"Missing keypoint {key}")
    return point


def _timestamp_ms(frame: dict[str, Any], fps: float) -> float:
    if "timestamp_ms" in frame:
        return float(frame["timestamp_ms"])
    return float(frame.get("frame_index", 0)) / max(fps, 1.0) * 1000.0


def _frame_observations(
    pose_frames: list[dict[str, Any]],
    ball_frames: dict[int, dict[str, Any]],
    fps: float,
    handedness: str,
) -> pd.DataFrame:
    shooting = "right" if handedness.lower().startswith("r") else "left"
    rows: list[dict[str, Any]] = []

    for frame in pose_frames:
        frame_index = int(frame["frame_index"])
        ball = ball_frames.get(frame_index)
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "timestamp_ms": _timestamp_ms(frame, fps),
            "ball_x": None,
            "ball_y": None,
            "ball_radius": None,
            "ball_confidence": None,
            "athlete_detected": True,
        }

        if ball:
            row["ball_x"] = float(ball["x"])
            row["ball_y"] = float(ball["y"])
            row["ball_radius"] = float(ball.get("radius", 0.0))
            row["ball_confidence"] = float(ball.get("confidence", 0.0))

        try:
            shoulder = _point(frame, f"{shooting}_shoulder")
            elbow = _point(frame, f"{shooting}_elbow")
            wrist = _point(frame, f"{shooting}_wrist")
            hip = _point(frame, f"{shooting}_hip")
            knee = _point(frame, f"{shooting}_knee")
            ankle = _point(frame, f"{shooting}_ankle")

            forearm_dx = wrist["x"] - elbow["x"]
            forearm_dy = wrist["y"] - elbow["y"]
            row["elbow_flexion"] = _angle(shoulder, elbow, wrist)
            row["knee_load"] = _angle(hip, knee, ankle)
            row["forearm_verticality"] = 90.0 - abs(math.degrees(math.atan2(forearm_dx, -forearm_dy)))
            row["elbow_flare"] = abs(math.degrees(math.atan2(elbow["x"] - shoulder["x"], abs(elbow["y"] - shoulder["y"]) + 1e-6)))
            row["release_height_px"] = abs(wrist["y"] - ankle["y"])
            row["hip_y"] = hip["y"]
            row["wrist_y"] = wrist["y"]
            if ball:
                row["ball_hand_distance_px"] = _distance({"x": ball["x"], "y": ball["y"]}, wrist)
        except KeyError:
            row["athlete_detected"] = False

        rows.append(row)

    return pd.DataFrame(rows)


def _segment_shots(frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame()

    shots: list[dict[str, Any]] = []
    active_start: int | None = None
    release_candidate: int | None = None
    min_gap_frames = 18

    for idx, row in frame_df.iterrows():
        hand_dist = row.get("ball_hand_distance_px")
        ball_seen = pd.notna(row.get("ball_x"))

        if active_start is None and ball_seen and pd.notna(hand_dist) and hand_dist < 48:
            active_start = int(row["frame_index"])
            release_candidate = None
            continue

        if active_start is not None and ball_seen and pd.notna(hand_dist):
            if hand_dist > 62 and release_candidate is None:
                release_candidate = int(row["frame_index"])

            if release_candidate is not None and int(row["frame_index"]) - release_candidate >= min_gap_frames:
                window = frame_df[
                    (frame_df["frame_index"] >= active_start) & (frame_df["frame_index"] <= int(row["frame_index"]))
                ]
                set_point_row = window.loc[window["wrist_y"].idxmin()] if "wrist_y" in window and window["wrist_y"].notna().any() else window.iloc[0]
                apex_row = window.loc[window["hip_y"].idxmin()] if "hip_y" in window and window["hip_y"].notna().any() else window.iloc[0]
                shots.append(
                    {
                        "shot_id": f"shot_{len(shots)+1:03d}",
                        "shot_start_frame": active_start,
                        "set_point_frame": int(set_point_row["frame_index"]),
                        "release_frame": release_candidate,
                        "shot_end_frame": int(row["frame_index"]),
                        "apex_frame": int(apex_row["frame_index"]),
                    }
                )
                active_start = None
                release_candidate = None

    return pd.DataFrame(shots)


def _extract_shot_features(
    shot_windows: pd.DataFrame,
    frame_df: pd.DataFrame,
    manifest: VideoManifest,
    athlete_profile: dict[str, Any],
    teacher_model: str,
    source_dataset: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    standing_reach = float(athlete_profile.get("standing_reach_m", 2.4))
    height = float(athlete_profile.get("height_m", 1.88))
    wingspan = float(athlete_profile.get("wingspan_m", 1.95))
    handedness = str(athlete_profile.get("handedness", "right"))

    for _, shot in shot_windows.iterrows():
        window = frame_df[
            (frame_df["frame_index"] >= shot["shot_start_frame"]) & (frame_df["frame_index"] <= shot["shot_end_frame"])
        ]
        if window.empty:
            continue

        set_point = window[window["frame_index"] == shot["set_point_frame"]].iloc[0]
        release = window[window["frame_index"] == shot["release_frame"]].iloc[0]
        release_time_ms = float(release["timestamp_ms"] - set_point["timestamp_ms"])
        release_vs_apex_ms = float(
            release["timestamp_ms"] - window[window["frame_index"] == shot["apex_frame"]].iloc[0]["timestamp_ms"]
        )
        release_height_ratio = float(release.get("release_height_px", 0.0)) / max(standing_reach * 100.0, 1.0)
        jump_height = max(0.15, min(0.7, abs(window["hip_y"].max() - window["hip_y"].min()) / max(height * 260.0, 1.0)))

        rows.append(
            {
                "athlete_id": athlete_profile.get("athlete_id", "unknown"),
                "shot_id": shot["shot_id"],
                "session_date": datetime.now(UTC).date().isoformat(),
                "fps": int(round(manifest.fps)),
                "side_video": Path(manifest.stored_path).name if manifest.view == "side" else "",
                "angle45_video": Path(manifest.stored_path).name if manifest.view == "angle45" else "",
                "shot_start_frame_side": int(shot["shot_start_frame"]) if manifest.view == "side" else None,
                "set_point_frame_side": int(shot["set_point_frame"]) if manifest.view == "side" else None,
                "release_frame_side": int(shot["release_frame"]) if manifest.view == "side" else None,
                "shot_end_frame_side": int(shot["shot_end_frame"]) if manifest.view == "side" else None,
                "shot_start_frame_45": int(shot["shot_start_frame"]) if manifest.view == "angle45" else None,
                "set_point_frame_45": int(shot["set_point_frame"]) if manifest.view == "angle45" else None,
                "release_frame_45": int(shot["release_frame"]) if manifest.view == "angle45" else None,
                "shot_end_frame_45": int(shot["shot_end_frame"]) if manifest.view == "angle45" else None,
                "make": None,
                "shot_type": "jump_shot",
                "distance_ft": None,
                "source_dataset": source_dataset,
                "source_tier": "teacher" if teacher_model != "manual_review" else "gold",
                "annotation_quality": "teacher" if teacher_model != "manual_review" else "manual",
                "teacher_model": teacher_model,
                "clip_uid": manifest.clip_id,
                "notes": f"processed_{manifest.view}",
                "handedness": handedness,
                "height_m": height,
                "wingspan_m": wingspan,
                "standing_reach_m": standing_reach,
                "release_time_ms_side": release_time_ms if manifest.view == "side" else None,
                "release_time_ms_45": release_time_ms if manifest.view == "angle45" else None,
                "paired_view_available": False,
                "release_timing_gap_ms": 0.0,
                "has_manual_stage_tags": False,
                "is_training_candidate": True,
                "elbow_flexion": float(release.get("elbow_flexion", set_point.get("elbow_flexion", 86.0))),
                "knee_load": float(window["knee_load"].min()) if window["knee_load"].notna().any() else 106.0,
                "forearm_verticality": float(release.get("forearm_verticality", 90.0)),
                "elbow_flare": float(release.get("elbow_flare", 4.0)),
                "release_height_ratio": release_height_ratio,
                "release_timing_ms": release_time_ms,
                "release_at_apex_offset_ms": release_vs_apex_ms,
                "jump_height": jump_height,
            }
        )

    return pd.DataFrame(rows)


def process_session(
    project_root: Path,
    manifest_path: Path,
    athlete_profile_path: Path,
    pose_json: Path | None,
    ball_json: Path | None,
    source_dataset: str,
    teacher_model: str,
) -> dict[str, Path]:
    manifest_data = json.loads(manifest_path.read_text())
    manifest = VideoManifest(**manifest_data)
    athlete_profile = json.loads(athlete_profile_path.read_text())
    pose_payload = _load_json(pose_json)
    ball_payload = _load_json(ball_json)

    pose_frames = pose_payload.get("frames", [])
    ball_frames = {int(frame["frame_index"]): frame for frame in ball_payload.get("frames", [])}
    if not pose_frames:
        raise RuntimeError("Pose frames are required to process a session. Supply a teacher-model pose JSON.")

    frame_df = _frame_observations(pose_frames, ball_frames, manifest.fps, athlete_profile.get("handedness", "right"))
    shot_df = _segment_shots(frame_df)
    shot_records = _extract_shot_features(
        shot_df,
        frame_df,
        manifest,
        athlete_profile,
        teacher_model=teacher_model,
        source_dataset=source_dataset,
    )

    dirs = _project_dirs(project_root)
    session_dir = dirs["processed"] / manifest.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    frame_parquet = session_dir / f"{manifest.clip_id}_frame_observations.parquet"
    shots_parquet = session_dir / f"{manifest.clip_id}_shot_records.parquet"
    session_json = session_dir / f"{manifest.clip_id}_session.json"

    frame_df.to_parquet(frame_parquet, index=False)
    shot_records.to_parquet(shots_parquet, index=False)
    session_json.write_text(
        json.dumps(
            {
                "manifest": asdict(manifest),
                "athlete_profile_path": str(athlete_profile_path.resolve()),
                "teacher_model": teacher_model,
                "source_dataset": source_dataset,
                "frame_count": int(len(frame_df)),
                "shot_count": int(len(shot_records)),
                "frame_observations_parquet": str(frame_parquet.resolve()),
                "shot_records_parquet": str(shots_parquet.resolve()),
            },
            indent=2,
        )
    )

    return {
        "frame_parquet": frame_parquet,
        "shots_parquet": shots_parquet,
        "session_json": session_json,
    }


def auto_process_session(
    project_root: Path,
    manifest_path: Path,
    athlete_profile_path: Path,
    source_dataset: str,
    teacher_model: str,
    frame_stride: int = 2,
) -> dict[str, Path]:
    teacher_outputs = run_builtin_teacher(
        project_root=project_root,
        manifest_path=manifest_path,
        athlete_profile_path=athlete_profile_path,
        frame_stride=frame_stride,
    )
    processed = process_session(
        project_root=project_root,
        manifest_path=manifest_path,
        athlete_profile_path=athlete_profile_path,
        pose_json=teacher_outputs["pose_json"],
        ball_json=teacher_outputs["ball_json"],
        source_dataset=source_dataset,
        teacher_model=teacher_model,
    )
    return {**teacher_outputs, **processed}


def auto_process_session_strong(
    project_root: Path,
    manifest_path: Path,
    athlete_profile_path: Path,
    source_dataset: str,
    teacher_model: str,
    frame_stride: int = 2,
    yolo_weights: str = "yolov8n.pt",
    pose_weights: str = "yolov8n-pose.pt",
    mediapipe_model: str | None = None,
) -> dict[str, Path]:
    teacher_outputs = run_strong_teacher(
        project_root=project_root,
        manifest_path=manifest_path,
        athlete_profile_path=athlete_profile_path,
        frame_stride=frame_stride,
        yolo_weights=yolo_weights,
        pose_weights=pose_weights,
        mediapipe_model=mediapipe_model,
    )
    processed = process_session(
        project_root=project_root,
        manifest_path=manifest_path,
        athlete_profile_path=athlete_profile_path,
        pose_json=teacher_outputs["pose_json"],
        ball_json=teacher_outputs["ball_json"],
        source_dataset=source_dataset,
        teacher_model=teacher_model,
    )
    return {**teacher_outputs, **processed}
