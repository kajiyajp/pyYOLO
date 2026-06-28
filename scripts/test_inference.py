# 作成: 2026-06-28
# 目的: 学習済みモデルで動画フレームに推論してバウンディングボックスを確認する

import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/train/l_mark_v1/weights/best.pt"
VIDEO_PATH = "samples/PointLock-mov.mp4"
OUTPUT_DIR = "samples/inference"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 均等に10フレーム抽出して推論
for i, pos in enumerate([int(total * t / 10) for t in range(10)]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    if len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            # バウンディングボックス描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 中心点
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            # 信頼度
            cv2.putText(frame, f"l_mark {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"frame{i:02d}: 検出OK 中心=({cx},{cy}) 信頼度={conf:.3f}")
    else:
        print(f"frame{i:02d}: 未検出")

    cv2.imwrite(f"{OUTPUT_DIR}/result_{i:02d}.jpg", frame)

cap.release()
print(f"\n結果を {OUTPUT_DIR}/ に保存しました")
