# 作成: 2026-06-28
# 目的: アノテーションが画像上の正しい位置に描かれているか確認する

import cv2

def draw_yolo_box(img, label_path):
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (int(cx * w), int(cy * h)), 5, (0, 0, 255), -1)
    return img

for i in range(5):
    img = cv2.imread(f"dataset/images/train/frame_{i:03d}.jpg")
    img = draw_yolo_box(img, f"dataset/labels/train/frame_{i:03d}.txt")
    cv2.imwrite(f"samples/check_{i:03d}.jpg", img)
    print(f"check_{i:03d}.jpg 保存")

print("samples/ フォルダを確認してください")
