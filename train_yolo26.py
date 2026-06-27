# 作成: 2026-06-28
# 目的: YOLO26nでl_markを学習し、YOLO11nと性能比較するためのモデルを作る

from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="l_mark_yolo26",
    project="runs/detect",
    degrees=15,
    scale=0.3,
    fliplr=0.5,
    flipud=0.3,
    patience=15,
    verbose=False,
)

print(f"mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
print(f"モデル: runs/detect/l_mark_yolo26/weights/best.pt")
