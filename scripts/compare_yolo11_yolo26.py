# 作成: 2026-06-28
# 目的: YOLO11n と YOLO26n の推論速度・精度を同じ動画フレームで比較する

import cv2
import time
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "samples/PointLock-mov.mp4"
OUTPUT_DIR = "samples/compare"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

models = {
    "yolo11n": YOLO("runs/detect/runs/train/l_mark_v2/weights/best.pt"),
    "yolo26n": YOLO("yolo26n.pt"),  # pretrained COCOモデル（l_mark未学習）
}

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for pos in [int(total * t / 5) for t in range(5)]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
cap.release()

print(f"{'モデル':<12} {'平均推論時間':>12} {'検出数':>8} {'最大信頼度':>12}")
print("-" * 50)

for name, model in models.items():
    times = []
    detections = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        results = model(frame, verbose=False, conf=0.1)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

        boxes = results[0].boxes
        n = len(boxes) if boxes is not None else 0
        max_conf = float(boxes.conf.max()) if n > 0 else 0.0
        detections.append((n, max_conf))

        # 結果画像を保存
        annotated = results[0].plot()
        cv2.imwrite(f"{OUTPUT_DIR}/{name}_frame{i:02d}.jpg", annotated)

    avg_time = np.mean(times)
    avg_det = np.mean([d[0] for d in detections])
    avg_conf = np.mean([d[1] for d in detections if d[0] > 0]) if any(d[0] > 0 for d in detections) else 0.0
    print(f"{name:<12} {avg_time:>10.1f}ms {avg_det:>8.1f} {avg_conf:>12.3f}")

print(f"\n結果画像: {OUTPUT_DIR}/")
print("※ yolo26n は l_mark 未学習（COCOモデル）のため検出数・信頼度は参考値")
