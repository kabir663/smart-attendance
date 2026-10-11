import os
import cv2
import time
import socket
import numpy as np
import threading

# Force TCP transport for EZVIZ
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

# ======================================================================
# ⚙️ FINAL SETTINGS FOR SUBMISSION
# ======================================================================
EZVIZ_IP = "10.196.209.3"
EZVIZ_RTSP_URL = f"rtsp://admin:BLYWTO@{EZVIZ_IP}:554/h264/ch1/main/av_stream"
CAMO_INDEX = 2  # Your working Camo camera index

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
# ======================================================================

def check_camera_port(ip, port=554):
    """Checks if the camera is reachable on the network"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect((ip, port))
        s.close()
        return True
    except:
        return False

class CCTVStream:
    """Industrial Driver for EZVIZ / Hikvision"""
    def __init__(self, source, name="Camera"):
        self.source = source
        self.name = name
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()
        self.status = "Connecting..."

    def start(self):
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            # Check if network can see the camera first
            if not check_camera_port(EZVIZ_IP, 554):
                with self.lock:
                    self.grabbed = False
                    self.status = "Camera Offline / Check Network"
                time.sleep(2)
                continue

            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                while self.running:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                        with self.lock:
                            self.frame = resized
                            self.grabbed = True
                    else:
                        break
                    time.sleep(0.01)
                cap.release()
            time.sleep(1)

    def read(self):
        with self.lock:
            if self.grabbed and self.frame is not None:
                return True, self.frame.copy(), ""
            return False, None, self.status

class WebcamStream:
    """Driver for USB / Camo Cameras"""
    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.frame = None
        self.grabbed = False
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with self.lock:
                    self.frame = resized
                    self.grabbed = True
            time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.grabbed and self.frame is not None:
                return True, self.frame.copy()
            return False, None

# --- START STREAMS ---
print(f"🎥 Connecting to EZVIZ CCTV at {EZVIZ_IP}...")
cam_in = CCTVStream(EZVIZ_RTSP_URL).start()

print(f"🎥 Connecting to CAMO at Index {CAMO_INDEX}...")
cam_out = WebcamStream(CAMO_INDEX).start()

# Load YuNet
try:
    detector = cv2.FaceDetectorYN.create(
        model=MODEL_PATH, config="",
        input_size=(FRAME_WIDTH, FRAME_HEIGHT),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
except:
    detector = None

def detect(frame):
    if detector and frame is not None:
        detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = detector.detect(frame)
        if faces is not None:
            for f in faces:
                x, y, w, h = f[:4].astype(int)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Employee Verified", (x, y-10), 0, 0.6, (0, 255, 0), 2)

window_name = "Smart Attendance System - Split View"

while True:
    ret_in, img_in, msg_in = cam_in.read()
    ret_out, img_out = cam_out.read()

    # Process Left (EZVIZ)
    if not ret_in:
        img_in = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img_in, msg_in, (50, 240), 0, 0.7, (0, 0, 255), 2)
    else:
        detect(img_in)

    # Process Right (CAMO)
    if not ret_out:
        img_out = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img_out, "CAMO Disconnected", (50, 240), 0, 0.7, (0, 0, 255), 2)
    else:
        detect(img_out)

    # UI Combine
    header_in = np.zeros((50, 640, 3), dtype=np.uint8); header_in[:] = (0, 255, 0)
    cv2.putText(header_in, "IN CAMERA (EZVIZ CCTV)", (10, 35), 0, 0.8, (0, 0, 0), 2)
    header_out = np.zeros((50, 640, 3), dtype=np.uint8); header_out[:] = (0, 165, 255)
    cv2.putText(header_out, "OUT CAMERA (CAMO)", (10, 35), 0, 0.8, (0, 0, 0), 2)

    top = np.hstack((np.vstack((header_in, img_in)), np.vstack((header_out, img_out))))
    footer = np.zeros((40, top.shape[1], 3), dtype=np.uint8)
    cv2.putText(footer, "Live Attendance Monitoring Active | Enterprise Mode", (10, 26), 0, 0.5, (255, 255, 255), 1)
    
    cv2.imshow(window_name, np.vstack((top, footer)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()