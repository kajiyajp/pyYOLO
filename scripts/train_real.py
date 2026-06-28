# 作成: 2026-06-28
# 目的: 実機撮影・注釈したdataset_realでYOLO11nを学習する
#       （撮影→注釈→学習の社内ループ実証）

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(
    data="configs/dataset_real.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    name="l_mark_real",
    project="runs/detect",
    degrees=15,
    scale=0.4,
    fliplr=0.5,
    flipud=0.3,
    hsv_v=0.4,
    patience=20,
    verbose=False,
)

print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print("モデル: runs/detect/l_mark_real/weights/best.pt")
