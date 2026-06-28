# 作成: 2026-06-28
# 目的: アノテーションツールで撮影・注釈したcaptures画像を
#       train/valに分割してYOLO学習用データセット(dataset_real)を作る

import os
import glob
import shutil
import random

# 撮影画像の場所（exe配下のcaptures）
SRC_DIR = "dist/pyYOLO-Annotator/captures"
DST = "dataset_real"
TRAIN_RATIO = 0.8

random.seed(42)

# 注釈(.txt)が存在する画像だけを対象にする
images = []
for jpg in sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg"))):
    txt = os.path.splitext(jpg)[0] + ".txt"
    if os.path.exists(txt) and os.path.getsize(txt) > 0:
        images.append(jpg)

print(f"注釈済み画像: {len(images)}枚")
random.shuffle(images)
split = int(len(images) * TRAIN_RATIO)
train_files = images[:split]
val_files = images[split:]
print(f"train: {len(train_files)}枚 / val: {len(val_files)}枚")

# 出力フォルダを作成
for s in ["train", "val"]:
    os.makedirs(f"{DST}/images/{s}", exist_ok=True)
    os.makedirs(f"{DST}/labels/{s}", exist_ok=True)

# 画像とラベルをコピー
for files, s in [(train_files, "train"), (val_files, "val")]:
    for jpg in files:
        name = os.path.basename(jpg)
        txt = os.path.splitext(jpg)[0] + ".txt"
        shutil.copy(jpg, f"{DST}/images/{s}/{name}")
        shutil.copy(txt, f"{DST}/labels/{s}/{os.path.splitext(name)[0]}.txt")

print(f"完了: {DST}/")
