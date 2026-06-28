# 作成: 2026-06-28
# 目的: 既存の bbox データセット(dataset_real)から、新規アノテーション無しで
#       YOLO Pose(Keypoint)用のデータセットを機械的に生成する「配線スモークテスト」。
#
#       【重要 — これは精度検証ではない】
#       キーポイントは bbox から機械的に導出する(箱の特定コーナー or 中心)だけで、
#       「真のL字頂点」ではない。目的はただ一つ:
#           pose 学習 → keypoint 予測 のパイプラインが
#           エラー無く最後まで回るか(配線確認)を今あるデータで即検証する。
#       ここで得られる mAP/keypoint座標の "良し悪し" には意味が無い。
#       見るべきは「学習が完走し、predict が (x, y, conf) を吐くか」だけ。
#
# 使い方:
#   python scripts/make_pose_smoketest.py
#       --src   C:/Users/kajiy/Documents/git-personal/pyYOLO/dataset_real
#       --dst   C:/Users/kajiy/Documents/git-personal/pyYOLO/dataset_pose_smoke
#       --mode  corner        # corner(箱の角=頂点と仮定) または center(箱中心)
#       --corner tl           # corner時の採用コーナー: tl/tr/bl/br/auto
#   (引数は全て省略可。省略時は上のデフォルトで動く)
#
# 出力:
#   <dst>/images/{train,val}/*.jpg        … 元画像をコピー
#   <dst>/labels/{train,val}/*.txt        … pose形式ラベル
#   <dst>/dataset_pose.yaml               … pose学習用 yaml(kpt_shape付き)
#   生成後、次のコマンドで学習が回ることを確認する:
#       yolo pose train data=<dst>/dataset_pose.yaml model=yolo11n-pose.pt epochs=10 imgsz=640

import argparse
import glob
import os
import shutil
import sys

# Windowsのコンソール(cp932)でも日本語が化けない/落ちないようにUTF-8へ
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------
# pose ラベルの仕様(Ultralytics):
#   1行 = 1インスタンス。
#   class  cx cy w h  px1 py1 v1  px2 py2 v2 ...
#   - cx cy w h : これまで通りの bbox(正規化)
#   - px,py     : キーポイント座標(正規化)
#   - v         : 可視フラグ(0=無し,1=隠れ,2=見える)。kpt_shape=[K,3]の時のみ付く。
#   今回は キーポイント数 K=1、次元 3(x,y,v) で作る。最小構成=配線確認に最適。
# ----------------------------------------------------------------------------

KPT_COUNT = 1   # キーポイント数(スモークなので1点だけ)
KPT_DIMS = 3    # x, y, visibility


def derive_keypoint(cx, cy, w, h, mode, corner):
    """bbox(正規化)から導出キーポイント(正規化)を1点返す。
    これは "真の頂点" ではなく、配線確認用の機械的な代用点。"""
    if mode == "center":
        return cx, cy
    # mode == "corner": 箱の四隅のいずれかを頂点と仮定する
    half_w, half_h = w / 2.0, h / 2.0
    corners = {
        "tl": (cx - half_w, cy - half_h),  # 左上
        "tr": (cx + half_w, cy - half_h),  # 右上
        "bl": (cx - half_w, cy + half_h),  # 左下
        "br": (cx + half_w, cy + half_h),  # 右下
    }
    if corner == "auto":
        # auto: とりあえず左上に固定(L字=箱の角の代用として一貫させるだけ)。
        # 真の頂点方向は画像ごとに違うので、ここでは "一貫した1コーナー" を選ぶ。
        corner = "tl"
    px, py = corners[corner]
    # 端で僅かにはみ出ても 0〜1 に丸めておく(学習側のバリデーション対策)
    px = min(1.0, max(0.0, px))
    py = min(1.0, max(0.0, py))
    return px, py


def convert_label(src_txt, dst_txt, mode, corner):
    """1つの bbox ラベルファイルを pose ラベルに変換して書き出す。
    戻り値: (処理した行数, スキップした行数)"""
    n_ok, n_skip = 0, 0
    out_lines = []
    with open(src_txt, "r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.split()
            if len(parts) < 5:
                n_skip += 1
                continue
            cls = parts[0]
            try:
                cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4]))
            except ValueError:
                n_skip += 1
                continue
            px, py = derive_keypoint(cx, cy, w, h, mode, corner)
            # 可視フラグ=2(見える)固定。導出点なので常に "可視" 扱いにする。
            vis = 2
            out_lines.append(
                f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
                f"{px:.6f} {py:.6f} {vis}"
            )
            n_ok += 1
    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
    return n_ok, n_skip


def copy_images_and_labels(src_root, dst_root, split, mode, corner):
    """1つの split(train/val)について画像コピー+ラベル変換を行う。"""
    img_src_dir = os.path.join(src_root, "images", split)
    lbl_src_dir = os.path.join(src_root, "labels", split)
    img_dst_dir = os.path.join(dst_root, "images", split)
    lbl_dst_dir = os.path.join(dst_root, "labels", split)
    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(lbl_dst_dir, exist_ok=True)

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(img_src_dir, ext)))
    images.sort()

    n_img, n_inst, n_skip, n_missing = 0, 0, 0, 0
    for img_path in images:
        name = os.path.basename(img_path)
        stem = os.path.splitext(name)[0]
        src_txt = os.path.join(lbl_src_dir, stem + ".txt")
        if not (os.path.exists(src_txt) and os.path.getsize(src_txt) > 0):
            # 注釈が無い画像はスモークでは飛ばす(配線確認に不要)
            n_missing += 1
            continue
        shutil.copy(img_path, os.path.join(img_dst_dir, name))
        ok, skip = convert_label(
            src_txt, os.path.join(lbl_dst_dir, stem + ".txt"), mode, corner
        )
        n_img += 1
        n_inst += ok
        n_skip += skip
    return n_img, n_inst, n_skip, n_missing


def write_pose_yaml(dst_root, yaml_path):
    """pose学習用 yaml を書き出す。kpt_shape と flip_idx が肝。"""
    # flip_idx: 左右反転augment時のキーポイント対応。1点なので [0] のみ。
    content = (
        "# 作成: 2026-06-28 (自動生成 by make_pose_smoketest.py)\n"
        "# 目的: bboxから導出した1点キーポイントで pose 配線を確認するための yaml。\n"
        "# 注意: kpt_shape=[1,3] は (キーポイント数=1, 次元=3[x,y,visible])。\n"
        "#       これは精度検証用ではなく、pose学習→predictが回るかの確認専用。\n"
        f"path: {dst_root.replace(os.sep, '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"kpt_shape: [{KPT_COUNT}, {KPT_DIMS}]\n"
        "flip_idx: [0]\n"
        "nc: 1\n"
        "names:\n"
        "  0: l_mark\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    ap = argparse.ArgumentParser(
        description="bbox dataset -> pose smoke-test dataset 生成"
    )
    ap.add_argument(
        "--src",
        default="C:/Users/kajiy/Documents/git-personal/pyYOLO/dataset_real",
        help="元の bbox データセットのルート(images/labels を含む)",
    )
    ap.add_argument(
        "--dst",
        default="C:/Users/kajiy/Documents/git-personal/pyYOLO/dataset_pose_smoke",
        help="生成する pose データセットのルート",
    )
    ap.add_argument(
        "--mode",
        default="corner",
        choices=["corner", "center"],
        help="キーポイントの導出法。corner=箱の角, center=箱中心",
    )
    ap.add_argument(
        "--corner",
        default="tl",
        choices=["tl", "tr", "bl", "br", "auto"],
        help="mode=corner の時に採用する隅",
    )
    args = ap.parse_args()

    src_root = os.path.abspath(args.src)
    dst_root = os.path.abspath(args.dst)

    if not os.path.isdir(os.path.join(src_root, "labels")):
        print(f"[ERROR] src に labels/ が見つからない: {src_root}", file=sys.stderr)
        sys.exit(1)

    print("=" * 64)
    print("pose スモークテスト用データセット生成")
    print("  これは『配線確認』であり精度検証ではない。")
    print(f"  src    : {src_root}")
    print(f"  dst    : {dst_root}")
    print(f"  mode   : {args.mode}  corner={args.corner}")
    print("=" * 64)

    total_img = 0
    total_inst = 0
    found_split = False
    for split in ("train", "val"):
        if not os.path.isdir(os.path.join(src_root, "images", split)):
            print(f"  [skip] {split} が無い")
            continue
        found_split = True
        n_img, n_inst, n_skip, n_missing = copy_images_and_labels(
            src_root, dst_root, split, args.mode, args.corner
        )
        total_img += n_img
        total_inst += n_inst
        print(
            f"  [{split}] 画像 {n_img}枚 / インスタンス {n_inst}個 "
            f"/ 注釈なしskip {n_missing}枚 / 不正行skip {n_skip}行"
        )

    # 空データセットのまま yaml を書くと「成功したのに中身が無い」誤解を招く。
    # train/val 構成が無い(=flatレイアウト等)か、注釈が1件も無い場合は明示エラーで止める。
    if total_img == 0:
        if not found_split:
            print(
                "[ERROR] src に images/{train,val} が見つかりません。\n"
                "        このスクリプトは train/val 分割済みデータセットを前提とします。\n"
                f"        確認してください: {os.path.join(src_root, 'images')}",
                file=sys.stderr,
            )
        else:
            print(
                "[ERROR] 注釈済み画像が0枚でした(全画像にラベルが無い/空)。\n"
                "        ラベル(labels/{train,val}/*.txt)を確認してください。",
                file=sys.stderr,
            )
        sys.exit(1)

    # yaml書き出し前に dst ルートを必ず作る(空でも write_pose_yaml が落ちないように)
    os.makedirs(dst_root, exist_ok=True)
    yaml_path = os.path.join(dst_root, "dataset_pose.yaml")
    write_pose_yaml(dst_root, yaml_path)

    print("-" * 64)
    print(f"合計: 画像 {total_img}枚 / インスタンス {total_inst}個")
    print(f"yaml : {yaml_path}")
    print("-" * 64)
    print("次に学習が回るか確認する(10エポックで十分):")
    print(
        f"  yolo pose train data={yaml_path} "
        f"model=yolo11n-pose.pt epochs=10 imgsz=640 batch=8"
    )
    print("学習後、predict が (x,y,conf) を吐くか確認する:")
    print(
        "  yolo pose predict model=runs/pose/train/weights/best.pt "
        f"source={dst_root}/images/val save=True"
    )
    print("")
    print("【結果の読み方 - 必読】")
    print("  ・見るべきは『完走するか』『predict が keypoint を返すか』だけ。")
    print("  ・mAP(pose) や keypoint の見た目の良し悪しは無意味。")
    print("    ラベルは bbox から作った代用点で、真の頂点ではないため。")
    print("  ・配線が通ったら、次は本物の頂点を少数枚だけ手動アノテして")
    print("    精度評価に進む(このスモークデータは精度評価に使わない)。")
    print("POSE_SMOKE_DONE")


if __name__ == "__main__":
    main()
