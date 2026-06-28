# 作成: 2026-06-28
# 目的: OBBモデルで動画フレームに推論して回転矩形を確認する

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs/obb/runs/obb/l_mark_obb/weights/best.pt"
VIDEO_PATH = "samples/PointLock-mov.mp4"
OUTPUT_DIR = "samples/obb_result"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i, pos in enumerate([int(total * t / 5) for t in range(5)]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame, verbose=False)
    obb = results[0].obb

    if obb is not None and len(obb) > 0:
        for j in range(len(obb)):
            conf = float(obb.conf[j])
            # OBBの4頂点を取得（xyxyxyxy形式）
            pts = obb.xyxyxyxy[j].cpu().numpy().reshape(4, 2).astype(int)
            # 回転矩形を描画
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            # 中心点
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            # 角度を算出して表示
            angle = float(obb.xywhr[j][4].cpu()) * 180 / np.pi
            cv2.putText(frame, f"OBB {conf:.2f} ang:{angle:.1f}deg",
                        (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"frame{i:02d}: 中心=({cx},{cy}) 信頼度={conf:.3f} 角度={angle:.1f}度")
    else:
        print(f"frame{i:02d}: 未検出")

    cv2.imwrite(f"{OUTPUT_DIR}/obb_{i:02d}.jpg", frame)

cap.release()
print(f"\n結果を {OUTPUT_DIR}/ に保存しました")
