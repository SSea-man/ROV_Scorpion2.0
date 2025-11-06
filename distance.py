import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = "C:\\Users\\uiu\\Downloads\\New folder\\runs\\content\\runs\\detect\\train\\weights\\best.pt"
#MODEL_PATH = "C:\\Users\\uiu\\Downloads\\New folder\\best (1).pt"


PIXEL_TO_MM = 0.5   
CONF_THRESH = 0.5   



model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()


drawing = False
start_point = None
end_point = None

def draw_line(event, x, y, flags, param):
    """Mouse drag line drawing + distance measurement."""
    global start_point, end_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

cv2.namedWindow("YOLO Distance Tool")
cv2.setMouseCallback("YOLO Distance Tool", draw_line)

print("[INFO] Click and drag to measure distance.")
print("[INFO] Press 'r' to reset or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame, stream=True, conf=CONF_THRESH)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            width_px = x2 - x1
            width_mm = width_px * PIXEL_TO_MM
            cv2.putText(frame, f"W: {width_mm:.1f} mm", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    
    if start_point and end_point:
        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.circle(frame, start_point, 5, (0, 255, 0), -1)
        cv2.circle(frame, end_point, 5, (0, 0, 255), -1)

        
        pixel_dist = np.linalg.norm(np.array(start_point) - np.array(end_point))
        mm_dist = pixel_dist * PIXEL_TO_MM
        cv2.putText(frame, f"Distance: {mm_dist:.2f} mm",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO Distance Tool", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        start_point, end_point = None, None
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()