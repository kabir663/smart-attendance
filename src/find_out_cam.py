import cv2
import os
import time

# Force TCP transport
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

HIKVISION_IP = "192.168.3.11"
PASSWORD = "Leeniv%4012"

print("==================================================")
print("🔍 LOBBY CAMERA FINDER - KEYFRAME WARMUP SCAN")
print("==================================================")
print("👉 Press SPACEBAR to view the NEXT camera.")
print("👉 Press Q to QUIT.\n")

# Active channels discovered on your NVR
ACTIVE_CHANNELS = list(range(1, 12)) 

for ch in ACTIVE_CHANNELS:
    # We test both Sub-stream (e.g. 102) and Main-stream (e.g. 101)
    for stream_type in ["2", "1"]:
        stream_code = f"{ch}0{stream_type}"
        stream_name = "Sub-Stream" if stream_type == "2" else "MAIN-STREAM (HD)"
        url = f"rtsp://admin:{PASSWORD}@{HIKVISION_IP}:554/Streaming/Channels/{stream_code}"
        
        print(f"🔄 Connecting to Channel {ch} [{stream_name} - Code: {stream_code}]...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            valid_frame = None
            
            # 🚨 Read 20 frames to allow H.265 keyframe to decode properly
            for _ in range(20):
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 5: # Not blank/black
                    valid_frame = frame
            
            cap.release()
            
            if valid_frame is not None:
                # Draw high-visibility header
                h, w = valid_frame.shape[:2]
                cv2.rectangle(valid_frame, (0, 0), (w, 80), (0, 0, 0), -1)
                cv2.putText(valid_frame, f"CHANNEL {ch} | {stream_name} (Code: {stream_code})", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(valid_frame, "Press SPACE = Next Camera | Press Q = Quit", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow("Hikvision Lobby Finder", valid_frame)
                print(f"  ✅ SHOWING FEED: Channel {ch} ({stream_name})")
                
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                
                if key in (ord('q'), ord('Q'), 27):
                    print("Scanner stopped.")
                    exit()
                
                # If we successfully showed a working image for this channel, jump to next channel
                break
            else:
                print(f"  ⚠️ Channel {ch} ({stream_name}) returned blank frames.")
        else:
            print(f"  ❌ Channel {ch} ({stream_name}) connection failed.")

cv2.destroyAllWindows()