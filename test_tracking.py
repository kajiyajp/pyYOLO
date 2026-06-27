# 作成: 2026-06-28
# 目的: YOLOv11のTracking機能で動画内のL字マークを追跡する

import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/train/l_mark_v2/weights/best.pt"
VIDEO_PATH = "samples/PointLock-mov.mp4"
OUTPUT_PATH = "samples/tracking_result.mp4"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_count = 0
detected_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # track: persist=True で前フレームの追跡を引き継ぐ
    results = model.track(frame, persist=True, conf=0.1, verbose=False)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # トラッキングID（追跡番号）
            track_id = int(box.id[0]) if box.id is not None else -1

            # 描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{track_id} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        detected_count += 1

    writer.write(frame)
    frame_count += 1

cap.release()
writer.release()

print(f"総フレーム数: {frame_count}")
print(f"検出フレーム数: {detected_count} ({detected_count/frame_count*100:.1f}%)")
print(f"結果動画: {OUTPUT_PATH}")
