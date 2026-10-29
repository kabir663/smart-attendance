import os
import time
import socket
import threading
import uuid
import pickle
import math
 
# ======================================================================
# 🚨 FORCE TCP TRANSPORT & ZERO-LATENCY FOR RTSP
# ======================================================================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
 
import cv2
import csv
import io
import glob
import sqlite3
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
 
app = Flask(__name__)
app.secret_key = "smart-attendance-secret"
 
# =========================
# PATHS
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "attendance.db")
EMP_DIR = os.path.join(BASE_DIR, "data", "employees")
VISITOR_DIR = os.path.join(BASE_DIR, "data", "visitors")
MODELS_DIR = os.path.join(BASE_DIR, "models")
 
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
 
os.makedirs(EMP_DIR, exist_ok=True)
os.makedirs(VISITOR_DIR, exist_ok=True)
 
# =========================
# CAMERA CONFIG  (1080p MAIN STREAMS)
# =========================
HIKVISION_IP = "192.168.3.11"
PASS = "Leeniv%4012"
 
def rtsp(ch_code: int) -> str:
    return f"rtsp://admin:{PASS}@{HIKVISION_IP}:554/Streaming/Channels/{ch_code}"
 
ALL_CAMERAS = [
    {"id": 1,  "name": "Cam 1",            "url": rtsp(101),  "role": "VIEW"},
    {"id": 2,  "name": "Cam 2",            "url": rtsp(201),  "role": "VIEW"},
    {"id": 3,  "name": "Cam 3",            "url": rtsp(301),  "role": "VIEW"},
    {"id": 4,  "name": "Cam 4",            "url": rtsp(401),  "role": "VIEW"},
    {"id": 5,  "name": "Cam 5",            "url": rtsp(501),  "role": "VIEW"},
    {"id": 6,  "name": "Reception (IN)",   "url": rtsp(601),  "role": "IN"},    # 1080p HD Main Stream
    {"id": 7,  "name": "Cam 7",            "url": rtsp(701),  "role": "VIEW"},
    {"id": 8,  "name": "Cam 8",            "url": rtsp(801),  "role": "VIEW"},
    {"id": 9,  "name": "Cam 9",            "url": rtsp(901),  "role": "VIEW"},
    {"id": 10, "name": "Lift Lobby (OUT)", "url": rtsp(1002), "role": "OUT"},   # Sub stream
    {"id": 11, "name": "Cam 11",           "url": rtsp(1101), "role": "VIEW"},
]
 
IN_CAMERA  = next(c["url"] for c in ALL_CAMERAS if c["role"] == "IN")
OUT_CAMERA = next(c["url"] for c in ALL_CAMERAS if c["role"] == "OUT")
 
# =========================
# ACCURACY & STABILITY CONFIG
# =========================
MATCH_THRESHOLD = 0.35          # SFace cosine similarity standard
REQUIRED_STABLE_FRAMES = 2      
COOLDOWN_SECONDS = 0            # Instant attendance logging
 
# =========================
# DB SETUP
# =========================
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
 
def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL, age INTEGER, department TEXT, photo_path TEXT, created_at TEXT)''')
           
    conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT, emp_id TEXT NOT NULL,
            name TEXT NOT NULL, date TEXT NOT NULL, in_time TEXT, out_time TEXT)''')
           
    conn.execute('''CREATE TABLE IF NOT EXISTS active_visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT, photo_path TEXT NOT NULL,
            feature BLOB NOT NULL, in_time TEXT NOT NULL, date TEXT NOT NULL)''')
           
    conn.commit()
    conn.close()
 
# =========================
# DYNAMIC THREADED CAMERA STREAM
# =========================
class ThreadedCameraStream:
    def __init__(self, cam_cfg, label=""):
        self.cam_cfg = cam_cfg
        self.label = label
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self.status = "Initializing..."
 
    def start(self):
        if self.running: return self
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        return self
 
    def _update(self):
        while self.running:
            cap = None
            try:
                if isinstance(self.cam_cfg, str):
                    cap = cv2.VideoCapture(self.cam_cfg, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                elif isinstance(self.cam_cfg, dict):
                    cap = cv2.VideoCapture(self.cam_cfg["index"], self.cam_cfg["backend"])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
 
                if cap and cap.isOpened():
                    while self.running:
                        grabbed = cap.grab()
                        if not grabbed:
                            with self.lock:
                                self.grabbed = False
                                self.status = f"{self.label} Reconnecting..."
                            break
                       
                        ret, frame = cap.retrieve()
                        if ret and frame is not None:
                            with self.lock:
                                self.frame = frame
                                self.grabbed = True
                        time.sleep(0.001)
                else:
                    with self.lock:
                        self.grabbed = False
                        self.status = f"{self.label} Connecting..."
            except Exception:
                with self.lock:
                    self.grabbed = False
                    self.status = f"{self.label} Network Error"
 
            if cap: cap.release()
            time.sleep(1)
 
    def read(self):
        with self.lock:
            if self.grabbed and self.frame is not None:
                return True, self.frame.copy(), ""
            return False, None, self.status
 
    def stop(self):
        self.running = False
 
# =========================
# FACE TRACKER & IDENTITY SMOOTHER
# =========================
class SimpleFaceTracker:
    def __init__(self, max_history=6):
        self.tracks = {}  # track_id -> {box, history, age}
        self.next_id = 1
        self.max_history = max_history
 
    def update(self, current_detections):
        updated_tracks = {}
        used_det_indices = set()
 
        for tid, trk in self.tracks.items():
            best_iou = 0.25  # IoU overlap threshold
            best_det_idx = -1
            tx, ty, tw, th = trk['box']
 
            for idx, det in enumerate(current_detections):
                if idx in used_det_indices: continue
                dx, dy, dw, dh = det['box']
               
                # Intersection over Union
                xA = max(tx, dx)
                yA = max(ty, dy)
                xB = min(tx + tw, dx + dw)
                yB = min(ty + th, dy + dh)
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = tw * th
                boxBArea = dw * dh
                iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
 
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx
 
            if best_det_idx != -1:
                det = current_detections[best_det_idx]
                used_det_indices.add(best_det_idx)
               
                # Exponential Moving Average for bounding box stabilization
                nx = int(0.65 * det['box'][0] + 0.35 * tx)
                ny = int(0.65 * det['box'][1] + 0.35 * ty)
                nw = int(0.65 * det['box'][2] + 0.35 * tw)
                nh = int(0.65 * det['box'][3] + 0.35 * th)
 
                trk['history'].append((det['label'], det['emp_id'], det['feature']))
                if len(trk['history']) > self.max_history:
                    trk['history'].pop(0)
 
                trk['box'] = (nx, ny, nw, nh)
                trk['age'] = 0
                updated_tracks[tid] = trk
 
        # Initialize new tracks
        for idx, det in enumerate(current_detections):
            if idx not in used_det_indices:
                tid = self.next_id
                self.next_id += 1
                updated_tracks[tid] = {
                    'box': det['box'],
                    'history': [(det['label'], det['emp_id'], det['feature'])],
                    'age': 0
                }
 
        self.tracks = updated_tracks
       
        # Majority voting identity smoothing
        smoothed = []
        for tid, trk in self.tracks.items():
            labels = [h[0] for h in trk['history']]
            known_labels = [l for l in labels if l not in ("Unknown", "Unknown Visitor")]
           
            if known_labels:
                final_label = max(set(known_labels), key=known_labels.count)
            else:
                final_label = max(set(labels), key=labels.count)
 
            emp_id = None
            feature = None
            for h in reversed(trk['history']):
                if h[0] == final_label:
                    emp_id = h[1]
                    feature = h[2]
                    break
 
            smoothed.append({
                'box': trk['box'],
                'label': final_label,
                'emp_id': emp_id,
                'feature': feature
            })
        return smoothed
 
# =========================
# PHOTO CAPTURE HELPER (LAPTOP WEBCAM ONLY)
# =========================
def capture_face_photo(save_path):
    cap = None
    window = "Capture Employee Photo | S=Save | Q=Cancel"
    captured = False
    last_frame = None
 
    try:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]:
            if backend is not None:
                cap = cv2.VideoCapture(0, backend)
            else:
                cap = cv2.VideoCapture(0)
               
            if cap and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None: break
                cap.release()
                cap = None
 
        if cap is None or not cap.isOpened():
            return False, "Laptop webcam could not be opened."
 
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: continue
 
            last_frame = frame.copy()
            preview = frame.copy()
            cv2.putText(preview, "ALIGN FACE - Press S to SAVE | Q to CANCEL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window, preview)
 
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                if last_frame is not None:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, last_frame)
                    captured = True
                break
            if key in (ord("q"), ord("Q"), 27): break
    except Exception as e:
        return False, f"Capture failed: {e}"
    finally:
        if cap is not None: cap.release()
        try: cv2.destroyWindow(window)
        except: pass
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
 
    if captured: return True, "Photo captured successfully"
    return False, "Capture cancelled"
 
# =========================
# FACE MODELS
# =========================
def load_detector_recognizer():
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.35, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
    return detector, recognizer
 
def build_known_faces():
    detector, recognizer = load_detector_recognizer()
    known = []
    conn = get_db()
    employees = conn.execute("SELECT emp_id, name, photo_path FROM employees WHERE photo_path != ''").fetchall()
    conn.close()
 
    for emp in employees:
        if not emp["photo_path"]: continue
        emp_folder = os.path.dirname(emp["photo_path"])
        if not os.path.exists(emp_folder): continue
       
        image_paths = glob.glob(os.path.join(emp_folder, "*.jpg")) + glob.glob(os.path.join(emp_folder, "*.png"))
        emp_features = []
 
        for img_p in image_paths:
            img = cv2.imread(img_p)
            if img is None: continue
           
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is None or len(faces) == 0: continue
 
            aligned = recognizer.alignCrop(img, faces[0])
            feature = recognizer.feature(aligned)
            emp_features.append(feature)
 
        if emp_features:
            # Multi-sample averaging: combines multiple photo samples into a single master profile
            avg_feature = np.mean(emp_features, axis=0)
            norm = np.linalg.norm(avg_feature)
            if norm > 0:
                avg_feature = avg_feature / norm
            known.append({"emp_id": emp["emp_id"], "name": emp["name"], "feature": avg_feature})
            print(f"👤 Loaded Master Profile: {emp['name']} ({emp['emp_id']}) from {len(emp_features)} sample(s)")
           
    return known, detector, recognizer
 
# =========================
# DB ATTENDANCE & VISITOR LOGIC
# =========================
def mark_in(emp_id, name):
    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    conn = get_db()
    row = conn.execute("SELECT id, in_time FROM attendance WHERE emp_id=? AND date=?", (emp_id, date)).fetchone()
    if row is None:
        conn.execute("INSERT INTO attendance (emp_id, name, date, in_time, out_time) VALUES (?, ?, ?, ?, ?)", (emp_id, name, date, time_str, None))
        conn.commit()
        res = f"CHECK-IN: {name}"
    else:
        res = None
    conn.close()
    return res
 
def mark_out(emp_id, name):
    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    conn = get_db()
    row = conn.execute("SELECT id, in_time, out_time FROM attendance WHERE emp_id=? AND date=?", (emp_id, date)).fetchone()
    if row is None:
        conn.execute("INSERT INTO attendance (emp_id, name, date, in_time, out_time) VALUES (?, ?, ?, ?, ?)", (emp_id, name, date, None, time_str))
        conn.commit()
        res = f"CHECK-OUT: {name}"
    elif row["out_time"] is None:
        conn.execute("UPDATE attendance SET out_time=? WHERE id=?", (time_str, row["id"]))
        conn.commit()
        res = f"CHECK-OUT: {name}"
    else:
        res = None
    conn.close()
    return res
 
def prepare_face_crop(frame, x, y, fw, fh):
    pad_x = max(20, int(fw * 0.30))
    pad_y = max(20, int(fh * 0.35))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + fw + pad_x)
    y2 = min(frame.shape[0], y + fh + pad_y)
    face_crop = frame[y1:y2, x1:x2].copy()
    if face_crop.size == 0:
        return None

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    min_dim = 120
    if face_crop.shape[0] < min_dim or face_crop.shape[1] < min_dim:
        scale = max(min_dim / float(face_crop.shape[0]), min_dim / float(face_crop.shape[1]))
        new_size = (max(1, int(face_crop.shape[1] * scale)), max(1, int(face_crop.shape[0] * scale)))
        face_crop = cv2.resize(face_crop, new_size, interpolation=cv2.INTER_LINEAR)

    return face_crop


def visitor_already_logged(feature, recognizer):
    if feature is None:
        return False
    conn = get_db()
    rows = conn.execute("SELECT feature FROM active_visitors").fetchall()
    conn.close()

    for row in rows:
        stored_feature = pickle.loads(row["feature"])
        score = recognizer.match(stored_feature, feature, cv2.FaceRecognizerSF_FR_COSINE)
        if score >= 0.78:
            return True
    return False


def save_unknown_visitor(face_crop, feature):
    if face_crop is None or feature is None:
        return False

    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    filename = f"visitor_{uuid.uuid4().hex[:8]}.jpg"
    photo_path = os.path.join(VISITOR_DIR, filename)

    clean_crop = cv2.resize(face_crop, (220, 220), interpolation=cv2.INTER_LINEAR)
    ok = cv2.imwrite(photo_path, clean_crop)
    if not ok:
        return False

    feature_blob = pickle.dumps(feature)
    conn = get_db()
    conn.execute("INSERT INTO active_visitors (photo_path, feature, in_time, date) VALUES (?, ?, ?, ?)",
                 (photo_path, feature_blob, time_str, date))
    conn.commit()
    conn.close()
    return True
 
def match_and_remove_visitor(feature, recognizer):
    conn = get_db()
    rows = conn.execute("SELECT id, photo_path, feature FROM active_visitors").fetchall()
   
    best_score = -1.0
    best_id = None
    best_path = None
   
    for row in rows:
        stored_feature = pickle.loads(row["feature"])
        score = recognizer.match(stored_feature, feature, cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_score = score
            best_id = row["id"]
            best_path = row["photo_path"]
           
    if best_score >= MATCH_THRESHOLD and best_id is not None:
        conn.execute("DELETE FROM active_visitors WHERE id=?", (best_id,))
        conn.commit()
        if best_path and os.path.exists(best_path):
            try: os.remove(best_path)
            except: pass
        conn.close()
        return True
       
    conn.close()
    return False
 
# =========================
# STABILIZED AI ENGINE
# =========================
def process_frame(frame, role, known, detector, recognizer, tracker, stable_frame_tracker, event_display_until, cooldown_tracker):
    if frame is None: return frame
    now_ts = datetime.now().timestamp()
    orig_h, orig_w = frame.shape[:2]
   
    # 960p detection pass
    DET_W = 960
    scale = DET_W / float(orig_w)
    det_h = int(orig_h * scale)
    small_frame = cv2.resize(frame, (DET_W, det_h), interpolation=cv2.INTER_LINEAR)
 
    detector.setInputSize((DET_W, det_h))
    _, faces = detector.detect(small_frame)
    raw_detections = []
 
    if faces is not None:
        for face in faces:
            x = int(face[0] / scale)
            y = int(face[1] / scale)
            fw = int(face[2] / scale)
            fh = int(face[3] / scale)
           
            if fw < 18 or fh < 18: continue
            aspect = fw / float(fh + 1e-6)
            if aspect < 0.45 or aspect > 1.55: continue
 
            face_full_res = face.copy()
            face_full_res[0], face_full_res[1] = x, y
            face_full_res[2], face_full_res[3] = fw, fh
            if len(face_full_res) >= 14:
                for i in range(4, 14, 2):
                    face_full_res[i] = face_full_res[i] / scale
                    face_full_res[i+1] = face_full_res[i+1] / scale
 
            aligned = recognizer.alignCrop(frame, face_full_res)
            feature = recognizer.feature(aligned)
 
            # Match evaluation with candidate margin check
            matches = []
            for person in known:
                score = recognizer.match(person["feature"], feature, cv2.FaceRecognizerSF_FR_COSINE)
                matches.append((person, score))
 
            matches.sort(key=lambda item: item[1], reverse=True)
 
            recognized = False
            best_person = None
            if matches and matches[0][1] >= MATCH_THRESHOLD:
                # Candidate margin test: ensure top match is clearly ahead of second match
                if len(matches) == 1 or (matches[0][1] - matches[1][1] >= 0.02):
                    recognized = True
                    best_person = matches[0][0]
 
            if recognized:
                raw_detections.append({
                    'box': (x, y, fw, fh),
                    'label': best_person['name'],
                    'emp_id': best_person['emp_id'],
                    'feature': feature
                })
            else:
                raw_detections.append({
                    'box': (x, y, fw, fh),
                    'label': "Unknown Visitor" if role in ["IN", "OUT"] else "Unknown",
                    'emp_id': None,
                    'feature': feature
                })
 
    # Temporal Smoothing & Box Stabilization Pass
    smoothed_detections = tracker.update(raw_detections)
    current_frame_recognized_ids = set()
 
    for det in smoothed_detections:
        x, y, fw, fh = det['box']
        label = det['label']
        emp_id = det['emp_id']
        feature = det['feature']
 
        if emp_id is not None and label not in ("Unknown", "Unknown Visitor"):
            # REGISTERED EMPLOYEE LOGIC
            if role in ["IN", "OUT"]:
                current_frame_recognized_ids.add(emp_id)
                key = f"{role}:{emp_id}"
                stable_frame_tracker[key] = stable_frame_tracker.get(key, 0) + 1
               
                if stable_frame_tracker[key] >= REQUIRED_STABLE_FRAMES:
                    msg = mark_in(emp_id, label) if role == "IN" else mark_out(emp_id, label)
                    if msg:
                        event_display_until[key] = now_ts + 4
                        cooldown_tracker[key] = now_ts
 
                color = (0, 255, 0) if role == "IN" else (0, 165, 255)
                if now_ts <= event_display_until.get(key, 0):
                    cv2.rectangle(frame, (x, y - 32), (x + 150, y), (0, 180, 0) if role == "IN" else (0, 140, 255), -1)
                    cv2.putText(frame, "CHECK-IN" if role == "IN" else "CHECK-OUT", (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255) if role=="IN" else (0,0,0), 2)
                    label += " | LOGGED"
            else:
                color = (255, 255, 0) # Cyan tracking on VIEW cams
        else:
            # UNKNOWN LOGIC
            if role in ["IN", "OUT"]:
                color = (0, 0, 255)
                label = "Unknown Visitor"
                key = f"{role}:Visitor_{x//50}_{y//50}"
                stable_frame_tracker[key] = stable_frame_tracker.get(key, 0) + 1
 
                if stable_frame_tracker[key] >= REQUIRED_STABLE_FRAMES:
                    if role == "IN" and feature is not None:
                        face_crop = prepare_face_crop(frame, x, y, fw, fh)
                        if face_crop is not None and not visitor_already_logged(feature, recognizer):
                            saved = save_unknown_visitor(face_crop, feature)
                            if saved:
                                label = "VISITOR LOGGED (IN)"
                            else:
                                label = "VISITOR DETECTED"
                        else:
                            label = "VISITOR ALREADY LOGGED"
                    elif role == "OUT" and feature is not None:
                        matched = match_and_remove_visitor(feature, recognizer)
                        label = "VISITOR EXITED" if matched else "UNKNOWN EXITED"
                   
                    event_display_until[key] = now_ts + 3
 
                if now_ts <= event_display_until.get(key, 0):
                    cv2.rectangle(frame, (x, y - 32), (x + 200, y), (0, 0, 200), -1)
                    cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            else:
                color = (0, 0, 255)
                label = "Unknown"
 
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.putText(frame, label, (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
 
    if role in ["IN", "OUT"]:
        for key in list(stable_frame_tracker.keys()):
            if key.startswith(role) and key.split(":")[1] not in current_frame_recognized_ids and "Visitor" not in key:
                stable_frame_tracker[key] = 0
 
    return frame
 
# =========================
# ROUTES (DASHBOARD & EMPLOYEES)
# =========================
@app.route("/")
def dashboard():
    conn = get_db()
    today = datetime.now().strftime("%Y-%m-%d")
   
    total_employees = conn.execute("SELECT COUNT(*) AS c FROM employees").fetchone()["c"]
    inside_records = conn.execute("SELECT name, emp_id, in_time FROM attendance WHERE date=? AND in_time IS NOT NULL AND out_time IS NULL", (today,)).fetchall()
    employees_inside = len(inside_records)
    employees_outside = total_employees - employees_inside
    visitors_inside = conn.execute("SELECT COUNT(*) AS c FROM active_visitors").fetchone()["c"]
    total_occupancy = employees_inside + visitors_inside
    recent = conn.execute("SELECT * FROM attendance ORDER BY id DESC LIMIT 8").fetchall()
    conn.close()
   
    return render_template("dashboard.html",
                           total_employees=total_employees, employees_inside=employees_inside,
                           employees_outside=employees_outside, visitors_inside=visitors_inside,
                           total_occupancy=total_occupancy, inside_list=inside_records,
                           recent=recent, today=today)
 
@app.route("/employees")
def employees():
    conn = get_db()
    rows = conn.execute("SELECT * FROM employees ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("employees.html", employees=rows)
 
@app.route("/photo/<int:emp_pk>")
def serve_photo(emp_pk):
    conn = get_db()
    emp = conn.execute("SELECT photo_path FROM employees WHERE id=?", (emp_pk,)).fetchone()
    conn.close()
    if emp and emp["photo_path"] and os.path.exists(emp["photo_path"]):
        return send_from_directory(os.path.dirname(emp["photo_path"]), os.path.basename(emp["photo_path"]))
    return "", 404
 
@app.route("/photo/<int:emp_pk>/<path:filename>")
def serve_employee_photo_file(emp_pk, filename):
    conn = get_db()
    emp = conn.execute("SELECT photo_path FROM employees WHERE id=?", (emp_pk,)).fetchone()
    conn.close()
    if emp and emp["photo_path"] and os.path.exists(os.path.join(os.path.dirname(emp["photo_path"]), filename)):
        return send_from_directory(os.path.dirname(emp["photo_path"]), filename)
    return "", 404
 
@app.route("/employees/edit/<int:emp_pk>", methods=["GET", "POST"])
def edit_employee(emp_pk):
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id=?", (emp_pk,)).fetchone()
    if not emp:
        conn.close()
        return redirect(url_for("employees"))
 
    if request.method == "POST":
        new_emp_id = request.form.get("emp_id", "").strip()
        new_name = request.form.get("name", "").strip()
        new_age = request.form.get("age", "").strip()
        new_dept = request.form.get("department", "").strip()
 
        old_safe_name = "_".join(emp["name"].split())
        old_folder = os.path.join(EMP_DIR, f"{emp['emp_id']}_{old_safe_name}")
        new_safe_name = "_".join(new_name.split())
        new_folder = os.path.join(EMP_DIR, f"{new_emp_id}_{new_safe_name}")
        photo_path = emp["photo_path"]
 
        if os.path.exists(old_folder) and old_folder != new_folder:
            os.rename(old_folder, new_folder)
            if photo_path and old_folder in photo_path:
                photo_path = photo_path.replace(old_folder, new_folder)
 
        try:
            conn.execute("UPDATE employees SET emp_id=?, name=?, age=?, department=?, photo_path=? WHERE id=?",
                         (new_emp_id, new_name, int(new_age) if new_age else None, new_dept, photo_path, emp_pk))
            conn.commit()
            conn.close()
            return redirect(url_for("employees"))
        except sqlite3.IntegrityError:
            conn.close()
            return redirect(url_for("edit_employee", emp_pk=emp_pk))
 
    photos = [f for f in os.listdir(os.path.dirname(emp["photo_path"])) if f.lower().endswith(('.jpg', '.png'))] if emp["photo_path"] else []
    conn.close()
    return render_template("edit_employee.html", emp=emp, photos=photos)
 
@app.route("/employees/delete_photo/<int:emp_pk>/<path:filename>", methods=["POST"])
def delete_employee_photo(emp_pk, filename):
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id=?", (emp_pk,)).fetchone()
    if emp and emp["photo_path"]:
        folder = os.path.dirname(emp["photo_path"])
        target_file = os.path.join(folder, filename)
        if os.path.exists(target_file):
            os.remove(target_file)
            remaining = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
            if not remaining: conn.execute("UPDATE employees SET photo_path='' WHERE id=?", (emp_pk,))
            else: conn.execute("UPDATE employees SET photo_path=? WHERE id=?", (os.path.join(folder, remaining[0]), emp_pk))
            conn.commit()
    conn.close()
    return redirect(url_for("edit_employee", emp_pk=emp_pk))
 
@app.route("/employees/import", methods=["POST"])
def import_employees():
    file = request.files.get("csv_file")
    if not file: return redirect(url_for("employees"))
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    reader = csv.DictReader(stream)
    conn = get_db()
    for row in reader:
        try:
            emp_id, name = row.get("emp_id", "").strip(), row.get("name", "").strip()
            age, department = row.get("age", "").strip() or None, row.get("department", "").strip()
            if emp_id and name:
                conn.execute("INSERT INTO employees (emp_id, name, age, department, photo_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                             (emp_id, name, age, department, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        except sqlite3.IntegrityError: pass
    conn.commit()
    conn.close()
    return redirect(url_for("employees"))
 
@app.route("/employees/capture/<int:emp_pk>")
def capture_existing(emp_pk):
    conn = get_db()
    emp = conn.execute("SELECT * FROM employees WHERE id=?", (emp_pk,)).fetchone()
    if not emp:
        conn.close()
        return redirect(url_for("employees"))
 
    safe_name = "_".join(emp["name"].split())
    emp_folder = os.path.join(EMP_DIR, f"{emp['emp_id']}_{safe_name}")
    os.makedirs(emp_folder, exist_ok=True)
    photo_path = os.path.join(emp_folder, f"{int(datetime.now().timestamp())}.jpg")
 
    ok, msg = capture_face_photo(photo_path)
    if ok:
        conn.execute("UPDATE employees SET photo_path=? WHERE id=?", (photo_path, emp_pk))
        conn.commit()
    conn.close()
    return redirect(url_for("edit_employee", emp_pk=emp_pk))
 
@app.route("/employees/add", methods=["GET", "POST"])
def add_employee():
    if request.method == "POST":
        emp_id = request.form.get("emp_id", "").strip()
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        department = request.form.get("department", "").strip()
        action = request.form.get("action", "capture_save")
 
        if not emp_id or not name: return redirect(url_for("add_employee"))
 
        safe_name = "_".join(name.split())
        emp_folder = os.path.join(EMP_DIR, f"{emp_id}_{safe_name}")
        os.makedirs(emp_folder, exist_ok=True)
        photo_path = os.path.join(emp_folder, "1.jpg")
 
        if action == "capture_save":
            ok, msg = capture_face_photo(photo_path)
            if not ok: return redirect(url_for("add_employee"))
        else: photo_path = ""
 
        try:
            conn = get_db()
            conn.execute("INSERT INTO employees (emp_id, name, age, department, photo_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (emp_id, name, int(age) if age else None, department, photo_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            conn.close()
            return redirect(url_for("employees"))
        except sqlite3.IntegrityError:
            return redirect(url_for("add_employee"))
    return render_template("add_employee.html")
 
@app.route("/employees/delete/<int:emp_pk>", methods=["POST"])
def delete_employee(emp_pk):
    conn = get_db()
    conn.execute("DELETE FROM employees WHERE id=?", (emp_pk,))
    conn.commit()
    conn.close()
    return redirect(url_for("employees"))
 
@app.route("/attendance")
def attendance():
    conn = get_db()
    rows = conn.execute("SELECT * FROM attendance ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("attendance.html", records=rows)
 
@app.route("/attendance/delete/<int:att_id>", methods=["POST"])
def delete_attendance(att_id):
    conn = get_db()
    conn.execute("DELETE FROM attendance WHERE id=?", (att_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("attendance"))
 
@app.route("/live")
def live_attendance_page():
    return render_template("live.html")
 
# =========================
# SMART DYNAMIC QUAD-VIEW ENGINE
# =========================
@app.route("/live/start")
def live_attendance_start():
    try:
        known, detector, recognizer = build_known_faces()
    except Exception as e:
        flash(f"Model error: {e}", "error")
        return redirect(url_for("dashboard"))
 
    stream_pool = {}
    trackers = {c["id"]: SimpleFaceTracker() for c in ALL_CAMERAS}
   
    in_cfg = next(c for c in ALL_CAMERAS if c["role"] == "IN")
    out_cfg = next(c for c in ALL_CAMERAS if c["role"] == "OUT")
    view_cfgs = [c for c in ALL_CAMERAS if c["role"] == "VIEW"]
 
    stream_pool[in_cfg["id"]] = ThreadedCameraStream(in_cfg["url"], in_cfg["name"]).start()
    stream_pool[out_cfg["id"]] = ThreadedCameraStream(out_cfg["url"], out_cfg["name"]).start()
 
    stable_frame_tracker = {}
    event_display_until = {}
    cooldown_tracker = {}
 
    TILE_W, TILE_H = 640, 360  
    HEADER_H = 32
    VIEWS_PER_PAGE = 2
    total_pages = math.ceil(len(view_cfgs) / VIEWS_PER_PAGE)
    page_idx = 0
 
    window_name = "Smart Quad-View  |  Press Spacebar to Shuffle"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
 
    def get_tile(cam_cfg):
        cid = cam_cfg["id"]
        if cid not in stream_pool:
            stream_pool[cid] = ThreadedCameraStream(cam_cfg["url"], cam_cfg["name"]).start()
 
        ret, frame, msg = stream_pool[cid].read()
        if not ret or frame is None:
            tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            cv2.putText(tile, msg or "Connecting...", (20, TILE_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            processed_hd = process_frame(frame, cam_cfg["role"], known, detector, recognizer,
                                         trackers[cid], stable_frame_tracker, event_display_until, cooldown_tracker)
            tile = cv2.resize(processed_hd, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
 
        if cam_cfg["role"] == "IN": bar_color = (0, 200, 0); title = f"IN  |  {cam_cfg['name']}"
        elif cam_cfg["role"] == "OUT": bar_color = (0, 165, 255); title = f"OUT |  {cam_cfg['name']}"
        else: bar_color = (60, 60, 60); title = f"VIEW |  {cam_cfg['name']}"
 
        cv2.rectangle(tile, (0, 0), (TILE_W, HEADER_H), bar_color, -1)
        cv2.putText(tile, title, (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return tile
 
    try:
        while True:
            active_views = view_cfgs[page_idx * VIEWS_PER_PAGE : (page_idx + 1) * VIEWS_PER_PAGE]
            active_ids = {in_cfg["id"], out_cfg["id"]}.union({c["id"] for c in active_views})
 
            for cid in list(stream_pool.keys()):
                if cid not in active_ids:
                    stream_pool[cid].stop()
                    del stream_pool[cid]
 
            t_in = get_tile(in_cfg)
            t_out = get_tile(out_cfg)
            t_v1 = get_tile(active_views[0]) if len(active_views) > 0 else np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
            t_v2 = get_tile(active_views[1]) if len(active_views) > 1 else np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
 
            row1 = np.hstack([t_in, t_out])
            row2 = np.hstack([t_v1, t_v2])
            grid = np.vstack([row1, row2])
 
            footer_h = 45
            footer = np.zeros((footer_h, grid.shape[1], 3), dtype=np.uint8)
            cv2.putText(footer, f"SPACEBAR = Cycle Views (Page {page_idx+1}/{total_pages})   |   Q = Quit   |   AI Tracks Everywhere, Attendance Logs on IN/OUT Only",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
           
            display = np.vstack([grid, footer])
            cv2.imshow(window_name, display)
 
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27): break
            elif key in (ord("n"), ord("N"), 32):
                page_idx = (page_idx + 1) % total_pages
 
    finally:
        for s in stream_pool.values():
            s.stop()
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)
 
    return redirect(url_for("dashboard"))
 
if __name__ == "__main__":
    init_db()
    print("🚀 Dashboard: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)