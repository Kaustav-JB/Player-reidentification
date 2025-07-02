from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("best.pt")

# Initialize Deep SORT with tuned parameters
tracker = DeepSort(
    max_age=90,
    n_init=3,
    max_cosine_distance=0.2,
    nn_budget=100              
)

cap = cv2.VideoCapture("D:/Assignments (New)/Stealth Mode/Assignment Materials/15sec_input_720p.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output_tracked.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        class_id = int(cls)
        w, h = x2 - x1, y2 - y1
        aspect_ratio = h / (w + 1e-6)

        if class_id in [1, 2 ,3] and conf > 0.6 and w > 40 and h > 70 and 1.2 < aspect_ratio < 4.0:
            detections.append(([x1, y1, w, h], conf, None))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()