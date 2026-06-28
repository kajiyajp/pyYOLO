# 作成: 2026-06-28
# 目的: YOLOv11n-OBBモデルでL字マークの回転対応検出を学習する

from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")

results = model.train(
    data="configs/dataset_obb.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="l_mark_obb",
    project="runs/obb",
    degrees=30,     # 回転augmentationを強めに
    scale=0.3,
    fliplr=0.5,
    flipud=0.3,
    patience=15,
    verbose=False,
)

print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"モデル: runs/obb/l_mark_obb/weights/best.pt")
