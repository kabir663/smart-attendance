import cv2

def test_backend(backend_name, backend_id):
    print(f"\n===== Testing backend: {backend_name} =====")
    found = False
    for i in range(0, 10):
        cap = cv2.VideoCapture(i, backend_id)
        if not cap.isOpened():
            print(f"Index {i}: ❌ not opened")
            continue

        # warmup
        for _ in range(8):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Index {i}: ⚠️ opened but no frame")
            cap.release()
            continue

        brightness = float(frame.mean())
        h, w = frame.shape[:2]
        print(f"Index {i}: ✅ frame OK | {w}x{h} | brightness={brightness:.1f}")

        # show preview
        cv2.putText(frame, f"{backend_name} index {i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"{backend_name} - Camera {i}", frame)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()

        cap.release()
        found = True

    if not found:
        print("No cameras returned frames on this backend.")

test_backend("DSHOW", cv2.CAP_DSHOW)
test_backend("MSMF", cv2.CAP_MSMF)

print("\nDone. Note which window shows your PHONE camera.")