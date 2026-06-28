# 作成: 2026-06-28
# 目的: YOLO11 Pose(キーポイント)でL字マークの頂点を直接検出する学習を行う。
#       矩形(bbox)検出と違い、各物体に「キーポイント=L字の頂点」を出力させる。
#       役割: YOLOが粗く頂点を当てる → 既存CV(detect_white_corner)が±0.1pxに追い込む。
# 使い方: python scripts/train_pose.py <data.yaml> <run_name> [epochs]
#   例:   python scripts/train_pose.py configs/pose_example.yaml l_mark_pose 120

import sys


def main():
    """コマンドライン引数を受けてYOLO11 Pose学習を実行する"""
    # 引数チェック（data.yaml と run_name は必須）
    if len(sys.argv) < 3:
        print("使い方: python scripts/train_pose.py <data.yaml> <run_name> [epochs]")
        sys.exit(1)

    data = sys.argv[1]                                   # pose用 dataset.yaml
    name = sys.argv[2]                                   # 実行名(出力フォルダ名)
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 120

    # ベースモデルはPose専用の事前学習重み(未取得ならultralyticsが自動DL)
    base = "yolo11n-pose.pt"

    from ultralytics import YOLO
    model = YOLO(base)

    # ============================================================
    # 【重要】左右反転(fliplr)について — 単一非対称頂点では必ず無効化する
    # ------------------------------------------------------------
    #   ultralyticsの左右反転は (1)画像とキーポイント座標を幾何的に左右反転し
    #   (2)その後 keypoints[:, flip_idx, :] で点を並べ替える、という2段階。
    #   flip_idx=[0](1点)では(2)の並べ替えが起きないが、(1)の幾何反転は実行される。
    #   その結果、例えば「箱の左上角」を指していた頂点が反転後は画像上の右上に来るのに、
    #   ラベルは依然「同じ頂点クラス」のまま学習され、検出器が『どの角か』を一貫学習できず
    #   頂点位置精度(本タスクの目的=サブpx前段)を直接劣化させる。
    #   → 左右対称な点ペアを持つ多点構成(例 flip_idx:[1,0,3,2])でない限り fliplr=0.0。
    #   上下反転(flipud)も同じ理由で無効。
    #
    #   モザイク/回転はultralyticsがキーポイント座標を追従させるので破綻はしないが、
    #   本プロジェクトはデータが少なく(数十枚)頂点が1点なので、過度な増強は
    #   唯一の角の位置をぼかす。mosaicは控えめ+終盤OFF、回転も小さめにする。
    # ============================================================
    results = model.train(
        data=data, epochs=epochs, imgsz=640, batch=8,
        name=name, project="runs/pose",
        degrees=10,        # 回転(基板向き対策)。頂点1点なので控えめ
        scale=0.4,         # 拡縮で距離変化に対応
        translate=0.1,     # 平行移動
        fliplr=0.0,        # 左右反転は無効(非対称な単一頂点では学習を歪める)
        flipud=0.0,        # 上下反転も無効(頂点対応が崩れる)
        hsv_v=0.4,         # 明るさ変動(ライブ照明ばらつき対策)
        mosaic=0.5,        # モザイクは控えめ(少データ+単一頂点のため)
        close_mosaic=10,   # 終盤10エポックはモザイクOFF(頂点位置を素のデータで締める)
        patience=20,       # 改善が止まったら早期終了
        verbose=False,
        # --- Pose固有の損失重み(必要に応じて調整) ---
        # pose:  キーポイント位置の損失重み(既定12.0)。頂点精度を上げたいなら増やす。
        # kobj:  キーポイント存在確率の損失重み(既定2.0)。
        pose=12.0,
        kobj=2.0,
    )

    # 出力されたbest.ptのパスを最後に明示(タブ側/呼び出し元が拾えるように)
    save_dir = getattr(results, "save_dir", None)
    print(f"BEST_PT={save_dir}/weights/best.pt" if save_dir else "BEST_PT=runs/pose")

    # 主要メトリクスも出す(Poseでは (P)=keypoint側の指標)
    rd = getattr(results, "results_dict", {}) or {}
    map_box = rd.get("metrics/mAP50(B)", "N/A")
    map_pose = rd.get("metrics/mAP50(P)", "N/A")
    print(f"mAP50(box)={map_box}  mAP50(pose)={map_pose}")
    print("TRAIN_DONE")


if __name__ == "__main__":
    main()
