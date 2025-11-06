import cv2
from ultralytics import YOLO

MODEL_PATH = "C:\\Users\\uiu\\Downloads\\New folder\\runs\\content\\runs\\detect\\train\\weights\\best.pt"

CONF_THRESH = 0.5


print("[INFO] Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access the webcam.")
    exit()

print("[INFO] Starting real-time detection. Press ESC to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    
    results = model(frame, stream=True, conf=CONF_THRESH)

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, "Object")

            
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + tw, y1), color, -1)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("YOLOv8 Custom Object Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
print("[INFO] Detection stopped.")
