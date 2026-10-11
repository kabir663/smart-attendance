import socket
import cv2
from concurrent.futures import ThreadPoolExecutor

VERIFICATION_CODE = "WYLBTO"

def get_my_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '192.168.1.100'
    finally:
        s.close()
    return ip

def check_ip(ip_str):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.3)
    result = sock.connect_ex((ip_str, 554)) # Check RTSP Port 554
    sock.close()
    if result == 0:
        return ip_str
    return None

print("🔎 Scanning your local Wi-Fi network for EZVIZ RTSP cameras...")
my_ip = get_my_ip()
subnet = ".".join(my_ip.split(".")[:3])
print(f"📡 Your PC IP: {my_ip} (Scanning subnet {subnet}.1 to {subnet}.254)\n")

candidate_ips = [f"{subnet}.{i}" for i in range(1, 255)]
found_ips = []

with ThreadPoolExecutor(max_workers=50) as executor:
    results = executor.map(check_ip, candidate_ips)
    for res in results:
        if res:
            found_ips.append(res)

if not found_ips:
    print("❌ No devices with RTSP Port 554 found on this subnet.")
    print("👉 Make sure your PC is connected to the SAME Wi-Fi network as the EZVIZ camera!")
else:
    print(f"🎯 Found {len(found_ips)} candidate device(s) on your network: {found_ips}\n")
    print("⚡ Testing video stream connection on found IP(s)...")

    for target_ip in found_ips:
        url = f"rtsp://admin:{VERIFICATION_CODE}@{target_ip}:554/h264/ch1/main/av_stream"
        print(f"Trying RTSP URL: {url}")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            # Warmup
            for _ in range(10): cap.read()
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                print(f"\n🎉 BINGO! EZVIZ Camera found at Local IP: {target_ip}")
                print(f"✅ Exact Working RTSP URL:\n{url}\n")
                cv2.imshow(f"EZVIZ Found at {target_ip}", frame)
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                cap.release()
                break
            else:
                print(f"⚠️ Connected to {target_ip}, but frame is blank. (Turn 'Image Encryption' OFF in EZVIZ App Settings!)")
        else:
            print(f"❌ Failed to connect stream on {target_ip}")
        cap.release()