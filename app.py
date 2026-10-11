import os
import time
import socket
import threading
import uuid
import pickle

# ======================================================================
# 🚨 FORCE TCP TRANSPORT FOR EZVIZ CCTV 
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
# CAMERA CONFIG 
# =========================
EZVIZ_IP = "10.196.209.3"
IN_CAMERA  = f"rtsp://admin:BLYWTO@{EZVIZ_IP}:554/h264/ch1/main/av_stream"
OUT_CAMERA = {"index": 2, "backend": cv2.CAP_MSMF}  # Camo Studio

# =========================
# ACCURACY & STABILITY CONFIG
# =========================
MATCH_THRESHOLD = 0.35          
REQUIRED_STABLE_FRAMES = 3      
COOLDOWN_SECONDS = 10           

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
            
    # Active unknown visitors currently inside
    conn.execute('''CREATE TABLE IF NOT EXISTS active_visitors (
            id INTEGER PRIMARY KEY AUTOINCREMENT, photo_path TEXT NOT NULL,
            feature BLOB NOT NULL, in_time TEXT NOT NULL, date TEXT NOT NULL)''')
            
    conn.commit()
    conn.close()

# =========================
# THREADED CAMERA STREAM
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
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if cap and cap.isOpened():
                    while self.running:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            with self.lock:
                                self.frame = frame
                                self.grabbed = True
                        else:
                            with self.lock:
                                self.grabbed = False
                                self.status = f"{self.label} Stream Reconnecting..."
                            break
                        time.sleep(0.005)
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
# PHOTO CAPTURE HELPER
# =========================
def capture_face_photo(save_path):
    cap = None
    window = "Capture Photo | S=Save | Q=Cancel"
    captured = False
    last_frame = None

    try:
        for cfg, name in [(OUT_CAMERA, "Camo Phone"), (IN_CAMERA, "EZVIZ Camera")]:
            if isinstance(cfg, str):
                cap = cv2.VideoCapture(cfg, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(cfg["index"], cfg["backend"])
            if cap and cap.isOpened():
                break

        if cap is None or not cap.isOpened():
            return False, "No camera available."

        print("📸 Capture window opened. Press S to save, Q to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret or frame is None: continue

            last_frame = frame.copy()
            preview = frame.copy()
            cv2.putText(preview, "Press S to SAVE | Q to CANCEL", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
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
    detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.5, 0.3, 5000)
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
        for img_p in image_paths:
            img = cv2.imread(img_p)
            if img is None: continue
            
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is None or len(faces) == 0: continue

            aligned = recognizer.alignCrop(img, faces[0])
            feature = recognizer.feature(aligned)
            known.append({"emp_id": emp["emp_id"], "name": emp["name"], "feature": feature})
            print(f"👤 Loaded Known Face: {emp['name']} ({emp['emp_id']})")
            
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
        print(f"🟢 CHECK-IN LOGGED -> {name} | Time: {time_str}")
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
        print(f"🟠 CHECK-OUT LOGGED -> {name} | Time: {time_str}")
        res = f"CHECK-OUT: {name}"
    elif row["out_time"] is None:
        conn.execute("UPDATE attendance SET out_time=? WHERE id=?", (time_str, row["id"]))
        conn.commit()
        print(f"🟠 CHECK-OUT UPDATED -> {name} | Time: {time_str}")
        res = f"CHECK-OUT: {name}"
    else:
        res = None
    conn.close()
    return res

# ----------------- VISITOR PHOTO LOGIC -----------------
def save_unknown_visitor(face_crop, feature):
    """IN Camera: Save unknown face to DB and disk"""
    date = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    filename = f"visitor_{uuid.uuid4().hex[:8]}.jpg"
    photo_path = os.path.join(VISITOR_DIR, filename)
    
    cv2.imwrite(photo_path, face_crop)
    feature_blob = pickle.dumps(feature)
    
    conn = get_db()
    conn.execute("INSERT INTO active_visitors (photo_path, feature, in_time, date) VALUES (?, ?, ?, ?)",
                 (photo_path, feature_blob, time_str, date))
    conn.commit()
    conn.close()
    print(f"⚠️ VISITOR ENTERED | Photo Saved: {filename}")

def match_and_remove_visitor(feature, recognizer):
    """OUT Camera: Match against saved visitors, delete if found"""
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
        print(f"✅ VISITOR EXITED | Matched Score={best_score:.2f} | Photo Deleted")
        return True
        
    conn.close()
    print(f"❌ Unknown Exiting (No Match Found in DB)")
    return False

# =========================
# VIDEO PROCESSING ENGINE
# =========================
def process_frame(frame, direction, known, detector, recognizer, stable_frame_tracker, event_display_until, cooldown_tracker):
    if frame is None: return frame
    now_ts = datetime.now().timestamp()
    h, w = frame.shape[:2]
    
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    current_frame_recognized_ids = set()

    if faces is not None:
        for face in faces:
            x, y, fw, fh = map(int, face[:4])
            if fw < 35 or fh < 35: continue

            aligned = recognizer.alignCrop(frame, face)
            feature = recognizer.feature(aligned)

            best_score = -1.0
            best = None
            for person in known:
                score = recognizer.match(person["feature"], feature, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best = person

            recognized = best is not None and best_score >= MATCH_THRESHOLD

            if recognized:
                emp_id = best["emp_id"]
                current_frame_recognized_ids.add(emp_id)
                key = f"{direction}:{emp_id}"
                stable_frame_tracker[key] = stable_frame_tracker.get(key, 0) + 1
                
                if stable_frame_tracker[key] >= REQUIRED_STABLE_FRAMES:
                    if now_ts - cooldown_tracker.get(key, 0) > COOLDOWN_SECONDS:
                        msg = mark_in(emp_id, best["name"]) if direction == "IN" else mark_out(emp_id, best["name"])
                        if msg:
                            event_display_until[key] = now_ts + 4
                            cooldown_tracker[key] = now_ts

                color = (0, 255, 0) if direction == "IN" else (0, 165, 255)
                label = best['name']
                if now_ts <= event_display_until.get(key, 0):
                    cv2.rectangle(frame, (x, y - 28), (x + 110, y), (0, 180, 0) if direction == "IN" else (0, 140, 255), -1)
                    cv2.putText(frame, "CHECK-IN" if direction == "IN" else "CHECK-OUT", (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255) if direction=="IN" else (0,0,0), 2)
                    label += " | LOGGED"
            else:
                # ----------------- VISITOR LOGIC -----------------
                key = f"{direction}:Visitor_{x//50}_{y//50}"
                stable_frame_tracker[key] = stable_frame_tracker.get(key, 0) + 1
                color = (0, 0, 255)
                label = "Unknown Visitor"

                if stable_frame_tracker[key] >= REQUIRED_STABLE_FRAMES:
                    cd_key = f"{direction}:GlobalVisitor"
                    if now_ts - cooldown_tracker.get(cd_key, 0) > COOLDOWN_SECONDS:
                        if direction == "IN":
                            pad = 20
                            y1, y2 = max(0, y - pad), min(frame.shape[0], y + fh + pad)
                            x1, x2 = max(0, x - pad), min(frame.shape[1], x + fw + pad)
                            face_crop = frame[y1:y2, x1:x2].copy()
                            save_unknown_visitor(face_crop, feature)
                            label = "VISITOR LOGGED (IN)"
                        else:
                            matched = match_and_remove_visitor(feature, recognizer)
                            label = "VISITOR EXITED" if matched else "UNKNOWN EXITED"
                        
                        event_display_until[key] = now_ts + 3
                        cooldown_tracker[cd_key] = now_ts

                if now_ts <= event_display_until.get(key, 0):
                    cv2.rectangle(frame, (x, y - 28), (x + 160, y), (0, 0, 200), -1)
                    cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
            cv2.putText(frame, label, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for key in list(stable_frame_tracker.keys()):
        if key.startswith(direction) and key.split(":")[1] not in current_frame_recognized_ids and "Visitor" not in key:
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
    
    # Who is inside right now (Employees)
    inside_records = conn.execute("SELECT name, emp_id, in_time FROM attendance WHERE date=? AND in_time IS NOT NULL AND out_time IS NULL", (today,)).fetchall()
    employees_inside = len(inside_records)
    employees_outside = total_employees - employees_inside
    
    # Who is inside right now (Unknown Visitors) -> Counts active rows
    visitors_inside = conn.execute("SELECT COUNT(*) AS c FROM active_visitors").fetchone()["c"]
    
    total_occupancy = employees_inside + visitors_inside
    recent = conn.execute("SELECT * FROM attendance ORDER BY id DESC LIMIT 8").fetchall()
    conn.close()
    
    return render_template("dashboard.html", 
                           total_employees=total_employees, 
                           employees_inside=employees_inside,
                           employees_outside=employees_outside,
                           visitors_inside=visitors_inside,
                           total_occupancy=total_occupancy,
                           inside_list=inside_records,
                           recent=recent, 
                           today=today)

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
        flash("Employee not found.", "error")
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
            flash(f"Updated profile for {new_name}.", "success")
            return redirect(url_for("employees"))
        except sqlite3.IntegrityError:
            conn.close()
            flash("Employee ID already belongs to another record.", "error")
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
            flash("Photo sample removed.", "success")
            remaining = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
            if not remaining:
                conn.execute("UPDATE employees SET photo_path='' WHERE id=?", (emp_pk,))
            else:
                new_main = os.path.join(folder, remaining[0])
                conn.execute("UPDATE employees SET photo_path=? WHERE id=?", (new_main, emp_pk))
            conn.commit()
    conn.close()
    return redirect(url_for("edit_employee", emp_pk=emp_pk))

@app.route("/employees/import", methods=["POST"])
def import_employees():
    file = request.files.get("csv_file")
    if not file or not file.filename.endswith('.csv'):
        flash("Please upload a valid .csv file.", "error")
        return redirect(url_for("employees"))

    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    reader = csv.DictReader(stream)

    conn = get_db()
    count = 0
    for row in reader:
        try:
            emp_id = row.get("emp_id", "").strip()
            name = row.get("name", "").strip()
            age = row.get("age", "").strip() or None
            department = row.get("department", "").strip()

            if emp_id and name:
                conn.execute(
                    "INSERT INTO employees (emp_id, name, age, department, photo_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (emp_id, name, age, department, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                )
                count += 1
        except sqlite3.IntegrityError:
            pass

    conn.commit()
    conn.close()
    flash(f"Successfully imported {count} employees from CSV!", "success")
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
    
    photo_name = f"{int(datetime.now().timestamp())}.jpg"
    photo_path = os.path.join(emp_folder, photo_name)

    ok, msg = capture_face_photo(photo_path)
    if ok:
        conn.execute("UPDATE employees SET photo_path=? WHERE id=?", (photo_path, emp_pk))
        conn.commit()
        flash(f"Face sample captured for {emp['name']}.", "success")
    else:
        flash(msg, "error")

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

        if not emp_id or not name:
            flash("Employee ID and Name are required.", "error")
            return redirect(url_for("add_employee"))

        safe_name = "_".join(name.split())
        emp_folder = os.path.join(EMP_DIR, f"{emp_id}_{safe_name}")
        os.makedirs(emp_folder, exist_ok=True)
        photo_path = os.path.join(emp_folder, "1.jpg")

        if action == "capture_save":
            ok, msg = capture_face_photo(photo_path)
            if not ok:
                flash(msg, "error")
                return redirect(url_for("add_employee"))
        else: photo_path = ""

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO employees (emp_id, name, age, department, photo_path, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (emp_id, name, int(age) if age else None, department, photo_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
            conn.close()
            flash(f"Employee {name} added successfully!", "success")
            return redirect(url_for("employees"))
        except sqlite3.IntegrityError:
            flash("Employee ID already exists.", "error")
            return redirect(url_for("add_employee"))
    return render_template("add_employee.html")

@app.route("/employees/delete/<int:emp_pk>", methods=["POST"])
def delete_employee(emp_pk):
    conn = get_db()
    conn.execute("DELETE FROM employees WHERE id=?", (emp_pk,))
    conn.commit()
    conn.close()
    flash(f"Employee deleted.", "success")
    return redirect(url_for("employees"))

# =========================
# ROUTES (ATTENDANCE & LIVE)
# =========================
@app.route("/attendance")
def attendance():
    conn = get_db()
    rows = conn.execute("SELECT * FROM attendance ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("attendance.html", records=rows)

@app.route("/attendance/delete/<int:att_id>", methods=["POST"])
def delete_attendance(att_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM attendance WHERE id=?", (att_id,)).fetchone()
    if row:
        conn.execute("DELETE FROM attendance WHERE id=?", (att_id,))
        conn.commit()
        flash(f"Deleted attendance entry for {row['name']}.", "success")
    conn.close()
    return redirect(url_for("attendance"))

# RESTORED MISSING ROUTE!
@app.route("/live")
def live_attendance_page():
    return render_template("live.html")

@app.route("/live/start")
def live_attendance_start():
    try: known, detector, recognizer = build_known_faces()
    except Exception as e:
        flash(f"Model error: {e}", "error")
        return redirect(url_for("dashboard"))

    cam_in = ThreadedCameraStream(IN_CAMERA, "EZVIZ").start()
    cam_out = ThreadedCameraStream(OUT_CAMERA, "CAMO").start()

    stable_frame_tracker = {}
    event_display_until = {}
    cooldown_tracker = {}
    PANEL_W, PANEL_H = 640, 480 

    try:
        while True:
            ret_i, frame_in, msg_in = cam_in.read()
            ret_o, frame_out, msg_out = cam_out.read()

            if not ret_i or frame_in is None:
                frame_in = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                cv2.putText(frame_in, msg_in, (30, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                frame_in = process_frame(frame_in, "IN", known, detector, recognizer, stable_frame_tracker, event_display_until, cooldown_tracker)

            if not ret_o or frame_out is None:
                frame_out = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                cv2.putText(frame_out, msg_out, (30, PANEL_H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                frame_out = process_frame(frame_out, "OUT", known, detector, recognizer, stable_frame_tracker, event_display_until, cooldown_tracker)

            frame_in = cv2.resize(frame_in, (PANEL_W, PANEL_H))
            frame_out = cv2.resize(frame_out, (PANEL_W, PANEL_H))
            
            cv2.rectangle(frame_in, (0, 0), (PANEL_W, 40), (0, 200, 0), -1)
            cv2.putText(frame_in, "IN CAMERA (CHECK-IN)", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            cv2.rectangle(frame_out, (0, 0), (PANEL_W, 40), (0, 165, 255), -1)
            cv2.putText(frame_out, "OUT CAMERA (CHECK-OUT)", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            combined = np.hstack([frame_in, frame_out])
            cv2.putText(combined, "Press Q to stop | Emergency Occupancy Engine Active", 
                        (16, combined.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            cv2.imshow("Smart Attendance - Realtime Split View", combined)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27): break
    finally:
        cam_in.stop()
        cam_out.stop()
        cv2.destroyAllWindows()
        for _ in range(5): cv2.waitKey(1)

    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    init_db()
    print("🚀 Dashboard: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False, threaded=True)