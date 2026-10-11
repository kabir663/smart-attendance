import cv2
import time

def test(index, backend, backend_name):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"{backend_name} index {index}: ❌ not opened")
        return

    # warmup
    for _ in range(10):
        cap.read()

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"{backend_name} index {index}: ⚠️ opened but no frame")
        cap.release()
        return

    brightness = float(frame.mean())
    h, w = frame.shape[:2]
    print(f"{backend_name} index {index}: ✅ {w}x{h} | brightness={brightness:.1f}")

    title = f"{backend_name} index {index}"
    cv2.putText(frame, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow(title, frame)
    cv2.waitKey(1500)
    cv2.destroyAllWindows()
    cap.release()

print("Scanning cameras... watch which window is Phone A and Phone B\n")

for i in range(0, 6):
    test(i, cv2.CAP_MSMF, "MSMF")

print("\n--- also testing DSHOW ---\n")
for i in range(0, 6):
    test(i, cv2.CAP_DSHOW, "DSHOW")

print("\nDone. Note indexes of your two phone cameras.")