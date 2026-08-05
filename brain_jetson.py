"""
Robot Brain — Jetson Orin Nano Edition
===================================
Optimized for NVIDIA Jetson Orin Nano:
  - jetson-inference DetectNet (TensorRT SSD-MobileNet-v2) for object detection
  - CUDA-accelerated YuNet face detection + SFace face recognition via OpenCV DNN
  - Frame-skip object detection (every DETECT_SKIP frames; boxes redrawn every frame)
  - GStreamer camera support for USB and CSI cameras
  - Twilio backend: read-only lookup by phone number (write disabled)

One-time setup:
  sudo nvpmodel -m 0
  sudo jetson_clocks
  pip install websockets

Ports:
  8765 — WebSocket  (brain events → dashboard)
  8766 — HTTP       (MJPEG camera stream → dashboard)
"""

import asyncio
import json
import logging
import subprocess
import threading
import time
import pickle
import urllib.request
import urllib.parse
import numpy as np
import websockets
import cv2
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_DEVICE  = "/dev/video0"   # EMEET SmartCam — /dev/video1 is its metadata node, not capture
USE_CSI_CAMERA = False
FRAME_W        = 640
FRAME_H        = 480

# ── Network ───────────────────────────────────────────────────────────────────
WS_HOST   = "localhost"
WS_PORT   = 8765
CAM_PORT  = 8766   # serves /video (MJPEG) and / (dashboard HTML)

# ── Face recognition ──────────────────────────────────────────────────────────
_HERE           = Path(__file__).parent
ENABLE_FACE     = True
PEOPLE_FILE     = str(_HERE / "people.json")
SFACE_MODEL     = str(_HERE / "face_recognition_sface_2021dec.onnx")
YUNET_MODEL     = str(_HERE / "face_detection_yunet_2023mar.onnx")
ENCODINGS_FILE  = str(_HERE / "face_encodings.pkl")
SFACE_THRESHOLD = 0.45

# ── Object detection (Ultralytics YOLO + TensorRT) ───────────────────────────
# Export once: python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='engine', half=True, device=0)"
_YOLO_ENGINE = _HERE.parent / "AI-OFFICE-ROBOT" / "yolov8n.pt"   # swap to yolov8n.engine after export
DETECT_MODEL     = str(_YOLO_ENGINE)
DETECT_SKIP      = 3                    # run inference every N frames; boxes redrawn every frame
DETECT_THRESHOLD = 0.5

# ── Twilio / Knack ────────────────────────────────────────────────────────────
TWILIO_FACE_URL   = "https://ai-robotics-2574.twil.io/airobotics-customer-lookup"
TWILIO_APPID      = "itexps_customer_lookup"
TWILIO_SECRET_KEY = "9d7e3c4a1f8b6e2d5c"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("brain")

# ── Shared camera frame ───────────────────────────────────────────────────────
latest_frame_lock  = threading.Lock()
latest_frame_bytes = None

# ── WebSocket state ───────────────────────────────────────────────────────────
ws_clients: set          = set()
ws_enrollment_handler    = None
enrollment_ref           = None


async def broadcast(event_type: str, text: str, extra: dict = None):
    payload = {
        "type":      event_type,
        "text":      text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    log.info("[%s] %s", event_type.upper(), text)
    if ws_clients:
        msg = json.dumps(payload)
        await asyncio.gather(
            *[ws.send(msg) for ws in list(ws_clients)],
            return_exceptions=True,
        )


async def ws_handler(websocket):
    ws_clients.add(websocket)
    log.info("Dashboard connected (%d total)", len(ws_clients))

    # Re-sync modal state for a freshly connected dashboard
    if enrollment_ref:
        resync = None
        if enrollment_ref.state == "asking":
            resync = {"type": "enrollment_ask", "text": "Unknown face — who is this?"}
        elif enrollment_ref.state == "awaiting_confirm":
            resync = {"type": "enrollment_found", "text": json.dumps({
                "name":  enrollment_ref.pending_name  or "",
                "phone": enrollment_ref.pending_phone or "",
                "email": enrollment_ref.pending_email or "",
            })}
        elif enrollment_ref.state == "awaiting_details":
            resync = {"type": "enrollment_new", "text": enrollment_ref.pending_phone or ""}
        if resync:
            try:
                resync["timestamp"] = datetime.now().isoformat(timespec="seconds")
                await websocket.send(json.dumps(resync))
            except Exception:
                pass

    try:
        async for raw in websocket:
            try:
                msg   = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "enroll_phone" and ws_enrollment_handler:
                    phone = msg.get("phone", "").strip()
                    if phone:
                        await ws_enrollment_handler("phone", phone=phone)

                elif mtype == "enroll_confirm" and ws_enrollment_handler:
                    await ws_enrollment_handler("confirm")

                elif mtype == "enroll_deny" and ws_enrollment_handler:
                    await ws_enrollment_handler("deny")

                elif mtype == "enroll_details" and ws_enrollment_handler:
                    name  = msg.get("name", "").strip()
                    email = msg.get("email", "").strip()
                    if name or email:
                        await ws_enrollment_handler("details", name=name, email=email)

                elif mtype == "enroll_dismiss":
                    if enrollment_ref and enrollment_ref.state not in ("capturing", "training"):
                        enrollment_ref.state          = "idle"
                        enrollment_ref._unknown_since = None

            except Exception as e:
                log.warning("Bad dashboard message: %s", e)
    finally:
        ws_clients.discard(websocket)


# ── MJPEG HTTP server ─────────────────────────────────────────────────────────

MJPEG_BOUNDARY = b"--frame"


_DASHBOARD_PATH = Path(__file__).parent / "dashboard_unified.html"


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    with latest_frame_lock:
                        frame = latest_frame_bytes
                    if frame is not None:
                        self.wfile.write(MJPEG_BOUNDARY + b"\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(1 / 24)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
        elif self.path in ("/", "/dashboard"):
            # Serve the dashboard HTML over HTTP so same-origin rules let it
            # reach the WebSocket and MJPEG stream without browser security blocks.
            try:
                html = _DASHBOARD_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            except Exception as exc:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(exc).encode())
        else:
            self.send_response(404)
            self.end_headers()


def start_mjpeg_server():
    HTTPServer.allow_reuse_address = True
    server = HTTPServer(("localhost", CAM_PORT), MJPEGHandler)
    log.info("Dashboard + MJPEG on http://localhost:%d/", CAM_PORT)
    server.serve_forever()


# ── People store ──────────────────────────────────────────────────────────────

def load_people() -> dict:
    p = Path(PEOPLE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def save_people(data: dict):
    Path(PEOPLE_FILE).write_text(json.dumps(data, indent=2))


def generate_face_id() -> str:
    date_str = datetime.now().strftime("%Y%m%d")
    counter  = len(PEOPLE) + 1
    return f"FACE_{date_str}_{counter:04d}"


PEOPLE = load_people()
reload_encodings_event = threading.Event()


# ── Enrollment state machine ──────────────────────────────────────────────────

class EnrollmentManager:
    STABLE_SECS    = 3.0
    SAMPLES_NEEDED = 30
    NO_FACE_GRACE  = 1.5

    def __init__(self):
        self.state          = "idle"
        self.pending_key    = None
        self.pending_name   = None
        self.pending_phone  = None
        self.pending_email  = None
        self.twilio_data    = {}     # populated if name found in Twilio
        self.frames         = []
        self._unknown_since = None
        self._no_face_since = None
        self._asking_at     = None


# ── Twilio lookup (read-only) ─────────────────────────────────────────────────

def lookup_by_phone(phone: str) -> dict:
    """GET customer profile from Twilio backend by phone number."""
    try:
        params = urllib.parse.urlencode({
            "appid":     TWILIO_APPID,
            "secretkey": TWILIO_SECRET_KEY,
            "phone":     phone,
        })
        with urllib.request.urlopen(f"{TWILIO_FACE_URL}?{params}", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("Twilio phone lookup failed for %s: %s", phone, exc)
        return {}


def lookup_by_name(name: str) -> dict:
    """GET customer profile from Twilio backend by name."""
    try:
        params = urllib.parse.urlencode({
            "appid":     TWILIO_APPID,
            "secretkey": TWILIO_SECRET_KEY,
            "name":      name,
        })
        with urllib.request.urlopen(f"{TWILIO_FACE_URL}?{params}", timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        log.warning("Twilio name lookup failed for %s: %s", name, exc)
        return {}


async def register_face_twilio(face_id: str, name: str, phone: str, email: str) -> dict:
    """POST a new face registration to the Twilio backend."""
    payload = json.dumps({
        "appid":     TWILIO_APPID,
        "secretkey": TWILIO_SECRET_KEY,
        "face_id":   face_id,
        "name":      name,
        "phone":     phone or "",
        "email":     email or "",
    }).encode("utf-8")

    def _post():
        req = urllib.request.Request(
            TWILIO_FACE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        result = await asyncio.get_event_loop().run_in_executor(None, _post)
        log.info("Twilio registered %s → %s", face_id, result)
        return result
    except Exception as exc:
        log.warning("Twilio register failed: %s", exc)
        return {}


# ── Camera + vision thread ────────────────────────────────────────────────────

def camera_thread(loop, broadcast_fn, enroll_mgr, enroll_done_fn):
    global latest_frame_bytes, PEOPLE

    # ── CUDA backend for OpenCV DNN (YuNet / SFace) ───────────────────────────
    _cuda_ok     = cv2.cuda.getCudaEnabledDeviceCount() > 0
    _dnn_backend = cv2.dnn.DNN_BACKEND_CUDA  if _cuda_ok else cv2.dnn.DNN_BACKEND_DEFAULT
    _dnn_target  = cv2.dnn.DNN_TARGET_CUDA   if _cuda_ok else cv2.dnn.DNN_TARGET_CPU
    log.info("OpenCV DNN: %s", "CUDA" if _cuda_ok else "CPU (CUDA not available)")

    # ── Face models ───────────────────────────────────────────────────────────
    sface_rec = cv2.FaceRecognizerSF.create(SFACE_MODEL, "", _dnn_backend, _dnn_target)
    yunet_det = cv2.FaceDetectorYN.create(
        YUNET_MODEL, "", (FRAME_W, FRAME_H), 0.5, 0.3, 5000, _dnn_backend, _dnn_target
    )
    known_encodings: dict = {}

    def load_recognizer():
        nonlocal known_encodings
        enc_path = Path(ENCODINGS_FILE)
        if enc_path.exists():
            with open(str(enc_path), "rb") as f:
                known_encodings = pickle.load(f)
            log.info("Face encodings loaded (%d people)", len(known_encodings))
        else:
            known_encodings = {}
            log.warning("No encodings file — face detection only")

    load_recognizer()

    # ── YOLO (GPU via CUDA; swap .pt → .engine for TensorRT after one-time export) ─
    yolo = YOLO(DETECT_MODEL)
    log.info("YOLO loaded: %s", DETECT_MODEL)

    # ── Camera open ───────────────────────────────────────────────────────────
    if USE_CSI_CAMERA:
        gst_str = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM),width={FRAME_W},height={FRAME_H},framerate=30/1 ! "
            f"nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
            f"video/x-raw,format=BGR ! appsink drop=1"
        )
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        log.info("Camera: CSI via GStreamer")
    else:
        cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))  # native hw compression
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)       # single-frame buffer reduces capture latency
        log.info("Camera: %s (MJPG %dx%d @ 30fps)", CAMERA_DEVICE, FRAME_W, FRAME_H)

    if not cap.isOpened():
        log.error("Could not open camera")
        asyncio.run_coroutine_threadsafe(
            broadcast_fn("error", "Camera not found — check index or CSI connection"), loop
        )
        return

    greeted         = set()
    last_seen       = None
    last_objects    = set()
    current_objects = set()
    last_det_boxes  = []      # cached boxes redrawn on skipped frames
    detect_count    = 0

    def emit(event_type, text):
        asyncio.run_coroutine_threadsafe(broadcast_fn(event_type, text), loop)

    while True:
        if reload_encodings_event.is_set():
            reload_encodings_event.clear()
            PEOPLE = load_people()
            load_recognizer()
            greeted.clear()
            emit("planning", "Face encodings reloaded")

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        h_frame, w_frame = frame.shape[:2]
        yunet_det.setInputSize((w_frame, h_frame))
        _, yunet_faces = yunet_det.detect(frame)

        current_seen = None
        any_unknown  = False

        # ── Select primary face: largest area (closest to camera) ─────────────
        # All faces are drawn; only the primary drives recognition & enrollment.
        primary_face = None
        if yunet_faces is not None and len(yunet_faces) > 0:
            cx, cy = w_frame / 2, h_frame / 2
            def _face_score(f):
                area = f[2] * f[3]
                fc_x = f[0] + f[2] / 2
                fc_y = f[1] + f[3] / 2
                dist = ((fc_x - cx) ** 2 + (fc_y - cy) ** 2) ** 0.5
                # Weight area heavily; penalise distance from centre
                return area - dist * 0.5
            primary_face = max(yunet_faces, key=_face_score)

            # Draw secondary faces (dim grey, no label)
            for face in yunet_faces:
                if face is not primary_face:
                    sx = max(0, int(face[0]))
                    sy = max(0, int(face[1]))
                    sw = min(int(face[2]), w_frame - sx)
                    sh = min(int(face[3]), h_frame - sy)
                    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (100, 100, 100), 1)

        # ── Recognise primary face ────────────────────────────────────────────
        if primary_face is not None:
            face = primary_face
            x = max(0, int(face[0]))
            y = max(0, int(face[1]))
            w = min(int(face[2]), w_frame - x)
            h = min(int(face[3]), h_frame - y)

            person_key, confidence = None, 0.0

            if known_encodings:
                try:
                    aligned    = sface_rec.alignCrop(frame, face)
                    query_feat = sface_rec.feature(aligned)
                    best_key, best_score = None, -1.0
                    for pk, feats in known_encodings.items():
                        for kf in feats:
                            score = float(sface_rec.match(
                                query_feat,
                                np.ascontiguousarray(kf, dtype=np.float32),
                                cv2.FaceRecognizerSF_FR_COSINE,
                            ))
                            if score > best_score:
                                best_score = score
                                best_key   = pk
                    if best_score >= SFACE_THRESHOLD:
                        person_key = best_key
                        confidence = round(best_score * 100, 1)
                    else:
                        confidence = round(max(0.0, best_score * 100), 1)
                except Exception as exc:
                    log.warning("SFace error: %s", exc)

            if person_key is not None:
                person = PEOPLE.get(person_key, {})
                name   = person.get("full_name", person_key)
                color  = (0, 220, 120)
                label  = f"{name} {confidence:.0f}%"
                current_seen = person_key
                if last_seen != person_key:
                    emit("face", f"Recognized: {name} ({confidence:.0f}%)")
                if person_key not in greeted:
                    greeted.add(person_key)
                    phone = person.get("phone", "")
                    if phone:
                        twilio_data = lookup_by_phone(phone)
                        if twilio_data:
                            emit("profile", json.dumps(twilio_data))
            else:
                color        = (60, 60, 220)
                label        = f"Unknown {confidence:.0f}%"
                current_seen = "unknown"
                any_unknown  = True
                if last_seen != "unknown":
                    emit("face", "Unknown face detected")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 28), (x + w, y), color, -1)
            cv2.putText(frame, label, (x + 6, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # ── Enrollment state machine ──────────────────────────────────────────
        if any_unknown:
            enroll_mgr._no_face_since = None
            if enroll_mgr._unknown_since is None:
                enroll_mgr._unknown_since = time.time()
            elif (time.time() - enroll_mgr._unknown_since >= EnrollmentManager.STABLE_SECS
                  and enroll_mgr.state == "idle"):
                enroll_mgr.state          = "asking"
                enroll_mgr._asking_at     = time.time()
                enroll_mgr._unknown_since = None
                asyncio.run_coroutine_threadsafe(
                    broadcast_fn("enrollment_ask", "Unknown face — who is this?"), loop
                )
        elif current_seen is None:
            # No face in frame at all
            if enroll_mgr._no_face_since is None:
                enroll_mgr._no_face_since = time.time()
            elif time.time() - enroll_mgr._no_face_since >= EnrollmentManager.NO_FACE_GRACE:
                enroll_mgr._unknown_since = None
                enroll_mgr._no_face_since = None
                # Cancel only if still in early steps — don't interrupt mid-capture
                if enroll_mgr.state in ("asking", "lookup", "awaiting_confirm", "awaiting_details"):
                    enroll_mgr.state  = "idle"
                    enroll_mgr.frames = []
                    emit("planning", "Face left — enrollment cancelled")
        else:
            # Known person in frame — reset unknown timer but keep no-face timer
            enroll_mgr._unknown_since = None

        # ── Face capture for enrollment (uses primary face) ───────────────────
        if enroll_mgr.state == "capturing":
            if primary_face is not None:
                try:
                    aligned = sface_rec.alignCrop(frame, primary_face)
                    feat    = sface_rec.feature(aligned).copy()
                    enroll_mgr.frames.append(feat)
                except Exception as e:
                    log.warning("Capture frame skipped: %s", e)
            progress = len(enroll_mgr.frames)
            banner = f"ENROLLING: {enroll_mgr.pending_name or '?'}  {progress}/{EnrollmentManager.SAMPLES_NEEDED}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (0, 160, 220), -1)
            cv2.putText(frame, banner, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if progress >= EnrollmentManager.SAMPLES_NEEDED:
                enroll_mgr.state = "training"
                enroll_done_fn()

        if last_seen is not None and current_seen is None:
            emit("face", "Face left camera view")
        last_seen = current_seen

        # ── Object detection (YOLO, every DETECT_SKIP frames) ────────────────
        detect_count += 1
        if detect_count % DETECT_SKIP == 0:
            results = yolo(frame, verbose=False, conf=DETECT_THRESHOLD)[0]
            current_objects = set()
            last_det_boxes  = []
            for box in results.boxes:
                lbl  = yolo.names[int(box.cls[0])]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                current_objects.add(lbl)
                last_det_boxes.append((x1, y1, x2, y2, lbl, conf))
            new_objs  = current_objects - last_objects
            gone_objs = last_objects - current_objects
            if new_objs:
                emit("objects", f"Appeared: {', '.join(sorted(new_objs))}")
            if gone_objs:
                emit("objects", f"Left: {', '.join(sorted(gone_objs))}")
            last_objects = current_objects

        for x1, y1, x2, y2, lbl, conf in last_det_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), (255, 165, 0), -1)
            cv2.putText(frame, f"{lbl} {conf:.0%}", (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with latest_frame_lock:
            latest_frame_bytes = jpeg.tobytes()


# ── Main async loop ───────────────────────────────────────────────────────────

async def run_brain():
    global enrollment_ref, ws_enrollment_handler

    enrollment     = EnrollmentManager()
    enrollment_ref = enrollment
    loop           = asyncio.get_event_loop()

    await broadcast("planning", "Robot brain started (Jetson Orin Nano)")

    def _open_dashboard():
        time.sleep(1.5)   # let HTTP + WebSocket servers finish binding first
        subprocess.Popen(
            ["xdg-open", f"http://localhost:{CAM_PORT}/"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    threading.Thread(target=_open_dashboard, daemon=True).start()

    async def handle_enroll(action: str, *, name: str = "", phone: str = "", email: str = ""):
        """Single entry point for all enrollment dashboard messages."""

        if action == "phone":
            # Step 1 — user entered their phone; look them up in Twilio
            if not phone or enrollment.state != "asking":
                return
            enrollment.pending_phone = phone
            enrollment.state         = "lookup"

            twilio_data = await asyncio.get_event_loop().run_in_executor(
                None, lookup_by_phone, phone
            )

            if twilio_data.get("found"):
                # Found in Twilio — populate profile and ask them to confirm
                enrollment.twilio_data   = twilio_data
                enrollment.pending_name  = (twilio_data.get("name")  or
                                            twilio_data.get("Name")  or "")
                enrollment.pending_key   = enrollment.pending_name.lower().replace(" ", "_")
                enrollment.pending_email = (twilio_data.get("email") or
                                            twilio_data.get("Email") or "")
                enrollment.state = "awaiting_confirm"
                await broadcast("enrollment_found", json.dumps({
                    "name":  enrollment.pending_name,
                    "phone": phone,
                    "email": enrollment.pending_email,
                }))
            else:
                # Not in system — ask for name + email
                enrollment.twilio_data = {}
                enrollment.state       = "awaiting_details"
                await broadcast("enrollment_new", phone)

        elif action == "confirm":
            # Step 2a — confirmed they are the Twilio profile
            if enrollment.state != "awaiting_confirm":
                return
            enrollment.state  = "capturing"
            enrollment.frames = []
            await broadcast("planning", f"Learning face for {enrollment.pending_name}…")

        elif action == "deny":
            # Step 2b — not the found profile; re-ask name
            enrollment.state        = "asking"
            enrollment.pending_name = None
            enrollment.pending_key  = None
            enrollment.twilio_data  = {}
            await broadcast("enrollment_ask", "Unknown face — who is this?")

        elif action == "details":
            # Step 2c — new user submitted name + email
            if enrollment.state != "awaiting_details":
                return
            enrollment.pending_name  = name
            enrollment.pending_key   = name.lower().replace(" ", "_")
            enrollment.pending_email = email
            enrollment.state         = "capturing"
            enrollment.frames        = []
            await broadcast("planning", f"Learning face for {name}…")

    ws_enrollment_handler = handle_enroll

    async def finish_enrollment():
        global PEOPLE
        name        = enrollment.pending_name
        phone       = enrollment.pending_phone
        email       = enrollment.pending_email or ""
        twilio_data = enrollment.twilio_data
        features    = enrollment.frames[:]

        def _reset():
            enrollment.state          = "idle"
            enrollment.pending_key    = None
            enrollment.pending_name   = None
            enrollment.pending_phone  = None
            enrollment.pending_email  = None
            enrollment.twilio_data    = {}
            enrollment.frames         = []
            enrollment._unknown_since = None
            enrollment._no_face_since = None

        if not features:
            await broadcast("planning", "Enrollment failed — no features captured")
            _reset()
            return

        key     = name.lower().replace(" ", "_")
        face_id = generate_face_id()

        # Prefer Twilio email over whatever was typed (Twilio is authoritative)
        if not email:
            email = (twilio_data.get("email") or twilio_data.get("Email") or "")

        try:
            enc_path = Path(ENCODINGS_FILE)
            enc_map: dict = {}
            if enc_path.exists():
                with open(str(enc_path), "rb") as f:
                    enc_map = pickle.load(f)
            enc_map[key] = features
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(enc_map, f)
        except Exception as exc:
            log.error("Failed to save encodings: %s", exc)
            await broadcast("planning", f"Enrollment save error: {exc}")
            _reset()
            return

        PEOPLE[key] = {
            "full_name":   name,
            "role":        "Visitor",
            "phone":       phone,
            "email":       email or None,
            "face_id":     face_id,
            "enrolled_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_people(PEOPLE)

        # Always POST — Twilio updates face_id for existing phone, creates new record otherwise
        await register_face_twilio(face_id, name, phone or "", email)

        reload_encodings_event.set()
        await broadcast("planning", f"Enrolled {name} ({len(features)} samples)")

        parts = [f"Name: {name}"]
        if phone:
            parts.append(f"Phone: {phone}")
        if email:
            parts.append(f"Email: {email}")
        await broadcast("profile", " | ".join(parts))
        _reset()

    def enroll_done():
        asyncio.run_coroutine_threadsafe(finish_enrollment(), loop)

    if ENABLE_FACE:
        t = threading.Thread(
            target=camera_thread,
            args=(loop, broadcast, enrollment, enroll_done),
            daemon=True,
        )
        t.start()
        await broadcast("planning", "Camera thread started")

    while True:
        await asyncio.sleep(3600)


async def main():
    mjpeg_thread = threading.Thread(target=start_mjpeg_server, daemon=True)
    mjpeg_thread.start()

    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, reuse_address=True)
    log.info("WebSocket on ws://%s:%d", WS_HOST, WS_PORT)

    await asyncio.gather(ws_server.serve_forever(), run_brain())


if __name__ == "__main__":
    asyncio.run(main())
