# 作成: 2026-06-28
# 目的: 引数で指定したデータセットでYOLOを学習する（学習・変換タブから呼ばれる）
# 使い方: python train_model.py <data.yaml> <run_name> [base_model] [epochs]

import sys

def main():
    """コマンドライン引数を受けてYOLO学習を実行する"""
    if len(sys.argv) < 3:
        print("使い方: python train_model.py <data.yaml> <run_name> [base.pt] [epochs]")
        sys.exit(1)
    data = sys.argv[1]
    name = sys.argv[2]
    base = sys.argv[3] if len(sys.argv) > 3 else "yolo11n.pt"
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 80

    from ultralytics import YOLO
    model = YOLO(base)
    results = model.train(
        data=data, epochs=epochs, imgsz=640, batch=8,
        name=name, project="runs/detect",
        degrees=15, scale=0.4, fliplr=0.5, flipud=0.3, hsv_v=0.4, patience=20,
        verbose=False,
    )
    # 出力されたbest.ptのパスを最後に明示（タブ側が拾えるように）
    save_dir = getattr(results, "save_dir", None)
    print(f"BEST_PT={save_dir}/weights/best.pt" if save_dir else "BEST_PT=runs/detect")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
