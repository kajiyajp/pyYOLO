# 作成: 2026-06-28
# 目的: 通常YOLO形式(cx cy w h)をOBB形式(4頂点)に変換する
# OBBラベル形式: class_id x1 y1 x2 y2 x3 y3 x4 y4 (正規化座標、左上→右上→右下→左下の順)

import os
import shutil

SRC_LABELS = ["dataset/labels/train", "dataset/labels/val"]
DST_BASE = "dataset_obb"

def yolo_to_obb(line):
    parts = line.strip().split()
    cls = parts[0]
    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    # 回転なし（angle=0）の4頂点
    x1, y1 = cx - w/2, cy - h/2  # 左上
    x2, y2 = cx + w/2, cy - h/2  # 右上
    x3, y3 = cx + w/2, cy + h/2  # 右下
    x4, y4 = cx - w/2, cy + h/2  # 左下

    return f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"

converted = 0
for src_labels_dir in SRC_LABELS:
    split = src_labels_dir.split("/")[-1]  # "train" or "val"
    src_images_dir = f"dataset/images/{split}"
    dst_labels_dir = f"{DST_BASE}/labels/{split}"
    dst_images_dir = f"{DST_BASE}/images/{split}"
    os.makedirs(dst_labels_dir, exist_ok=True)
    os.makedirs(dst_images_dir, exist_ok=True)

    for fname in os.listdir(src_labels_dir):
        if not fname.endswith(".txt"):
            continue
        src_path = f"{src_labels_dir}/{fname}"
        dst_path = f"{dst_labels_dir}/{fname}"
        with open(src_path) as f:
            lines = f.readlines()
        with open(dst_path, "w") as f:
            for line in lines:
                if line.strip():
                    f.write(yolo_to_obb(line) + "\n")

        # 対応する画像もコピー
        img_name = fname.replace(".txt", ".jpg")
        src_img = f"{src_images_dir}/{img_name}"
        if os.path.exists(src_img):
            shutil.copy(src_img, f"{dst_images_dir}/{img_name}")
        converted += 1

print(f"変換完了: {converted}件")
print(f"出力先: {DST_BASE}/")
