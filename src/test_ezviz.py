import os
import socket
import cv2

# Set short timeout so OpenCV doesn't freeze for 30 seconds
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000"

VERIFICATION_CODE = "BLYWTO"

# Common EZVIZ URL patterns
URL_PATTERNS = [
    "/h264/ch1/main/av_stream",
    "/h264/ch1/sub/av_stream",
    "/LiveMedia/ch1/main/Video",
    "/ch1/main",
    ""
]

def scan_network_for_rtsp():
    print("🔍 Step 1: Scanning local network for open EZVIZ cameras (Port 554)...")
    
    # Get PC local subnet (e.g., 192.168.1)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        pc_ip = s.getsockname()[0]
    except Exception:
        pc_ip = '192.168.1.100'
    finally:
        s.close()

    subnet = ".".join(pc_ip.split(".")[:3])
    print(f"💻 PC IP Address: {pc_ip} | Searching on subnet: {subnet}.X\n")

    found_ips = []
    # Test active IPs from arp output first, then scan range
    candidate_ips = ["192.168.1.105", "192.168.1.93", "192.168.1.94", "192.168.1.254"]
    candidate_ips += [f"{subnet}.{i}" for i in range(1, 255) if f"{subnet}.{i}" not in candidate_ips]

    for ip in candidate_ips:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.15)
        result = sock.connect_ex((ip, 554))
        if result == 0:
            print(f"  👉 Found active RTSP device at IP: {ip}")
            found_ips.append(ip)
        sock.close()

    return found_ips

def test_rtsp_urls(found_ips):
    print("\n🔍 Step 2: Testing EZVIZ RTSP streams on discovered IPs...")
    
    working_url = None
    for ip in found_ips:
        for pattern in URL_PATTERNS:
            url = f"rtsp://admin:{VERIFICATION_CODE}@{ip}:554{pattern}"
            print(f"  Testing: {url} ... ", end="", flush=True)
            
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    print("✅ SUCCESS! Video frame captured!")
                    working_url = url
                    return working_url
                else:
                    print("❌ Connected, but no video frame.")
            else:
                print("❌ Failed.")

    return None

if __name__ == "__main__":
    active_ips = scan_network_for_rtsp()
    if not active_ips:
        print("\n❌ No device found listening on RTSP Port 554.")
        print("💡 Check: Is the EZVIZ camera plugged into power and connected to the network?")
    else:
        url = test_rtsp_urls(active_ips)
        if url:
            print("\n" + "="*60)
            print(f"🎉 COPY THIS EXACT WORKING URL TO camera_feed.py:")
            print(f"   EZVIZ_URL = \"{url}\"")
            print("="*60)
        else:
            print("\n❌ Could not grab stream. Check if Image Encryption is OFF in the EZVIZ mobile app.")