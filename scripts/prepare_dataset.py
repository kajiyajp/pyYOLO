# 作成: 2026-06-28
# 目的: 画像とラベルをtrain/valに分割してYOLO学習用データセットを準備する

import os
import shutil
import random

IMAGES_DIR = "dataset/images/all"
LABELS_DIR = "dataset/labels/all"
TRAIN_RATIO = 0.8  # 80%を学習用、20%を検証用

random.seed(42)

# ファイル一覧取得
images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
random.shuffle(images)

split = int(len(images) * TRAIN_RATIO)
train_files = images[:split]
val_files = images[split:]

print(f"train: {len(train_files)}枚 / val: {len(val_files)}枚")

# フォルダ作成
for split_name in ["train", "val"]:
    os.makedirs(f"dataset/images/{split_name}", exist_ok=True)
    os.makedirs(f"dataset/labels/{split_name}", exist_ok=True)

# ファイルをコピー
for files, split_name in [(train_files, "train"), (val_files, "val")]:
    for img_file in files:
        label_file = img_file.replace(".jpg", ".txt")
        shutil.copy(f"{IMAGES_DIR}/{img_file}", f"dataset/images/{split_name}/{img_file}")
        label_path = f"{LABELS_DIR}/{label_file}"
        if os.path.exists(label_path):
            shutil.copy(label_path, f"dataset/labels/{split_name}/{label_file}")

print("分割完了")
