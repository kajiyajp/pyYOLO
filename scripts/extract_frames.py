# 作成: 2026-06-28
# 目的: 動画から均等にフレームを抽出してアノテーション用画像を準備する

import cv2
import os

VIDEO_PATH = "samples/PointLock-mov.mp4"
OUTPUT_DIR = "dataset/images/all"
TARGET_COUNT = 100  # 抽出枚数

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"総フレーム数: {total_frames}, FPS: {fps:.1f}, 長さ: {total_frames/fps:.1f}秒")
print(f"抽出枚数: {TARGET_COUNT}")

step = total_frames / TARGET_COUNT
saved = 0

for i in range(TARGET_COUNT):
    frame_pos = int(i * step)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if not ret:
        continue
    filename = f"{OUTPUT_DIR}/frame_{i:03d}.jpg"
    cv2.imwrite(filename, frame)
    saved += 1

cap.release()
print(f"{saved}枚を {OUTPUT_DIR}/ に保存しました")
