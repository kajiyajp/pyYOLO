# 作成: 2026-06-28
# 目的: 学習済みYOLO Pose(Keypoint)モデルの「頂点pixel誤差(精度)」と
#       「ライブ映像のジッター(安定性)・検出継続率」を計測する。
#
# 評価の主眼:
#   - accuracy: val画像で pred keypoint vs GT keypoint の pixel距離(平均/中央値/最大)。
#   - stability: 連続フレーム/動画で同一頂点のフレーム間標準偏差(ジッター)と検出継続率。
#
# 使い方:
#   # 精度(val画像 + GTラベル):
#   python scripts/measure_keypoint.py accuracy --model runs/pose/.../best.pt --data configs/dataset_pose.yaml
#   python scripts/measure_keypoint.py accuracy --model best.pt --images dataset_pose/images/val --labels dataset_pose/labels/val
#
#   # 安定性(動画):
#   python scripts/measure_keypoint.py stability --model best.pt --video samples/PointLock-mov.mp4
#
# 重要メモ(letterbox):
#   ultralytics の results[0].keypoints.xy は「元画像のpixel座標」で返る
#   (letterbox の scale/pad はライブラリ内部で復元済み)。
#   よって自前のletterbox逆変換は不要。GT側だけ「正規化 → 元画像pixel」に戻せばよい。

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------------ #
# 共通ユーティリティ
# ------------------------------------------------------------------ #

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _img2label_dir(img_dir):
    """images/... のパス『セグメント』を labels/... に置換して labelsディレクトリを得る。
    部分文字列置換(replace)だと親フォルダ名に 'images' を含むパスを誤置換するため、
    ultralytics の img2label_paths と同様に『パス成分単位』で最後の images を入れ替える。"""
    parts = list(Path(img_dir).parts)
    # 末尾側から最初に見つかった 'images' 成分を 'labels' に置換
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    # 'images' 成分が無い場合は従来挙動(後方互換)にフォールバック
    return Path(str(img_dir).replace("images", "labels", 1))


def load_yaml_paths(data_yaml):
    """dataset.yaml から val画像ディレクトリ・labelsディレクトリ・kpt形状を推定する。
    戻り値: (img_dir, lbl_dir, n_kpt, ndim)。ndim は各点の要素数(2 or 3)。"""
    import yaml  # ultralytics依存で同梱されている
    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", "."))
    val = cfg.get("val", "images/val")
    img_dir = (root / val) if not Path(val).is_absolute() else Path(val)
    # YOLO規約: images/... と同じ並びで labels/... を探す(セグメント単位で置換)
    lbl_dir = _img2label_dir(img_dir)
    n_kpt, ndim = None, None
    if "kpt_shape" in cfg:  # 例: [4, 3] → 4頂点, D=3
        ks = cfg["kpt_shape"]
        n_kpt = int(ks[0])
        if len(ks) > 1:
            ndim = int(ks[1])
    return img_dir, lbl_dir, n_kpt, ndim


def parse_gt_label(label_path, img_w, img_h, n_kpt=None, ndim=None):
    """
    GTラベル(YOLO pose形式)を読み、各インスタンスのkeypoint pixel座標を返す。

    1行 = "cls cx cy w h  px1 py1 v1  px2 py2 v2 ..."（全て0..1正規化, vは可視フラグ)
    戻り値: [ {"bbox": (x1,y1,x2,y2)px, "kpts": ndarray(K,2)px, "vis": ndarray(K,)} , ... ]

    step(各点の要素数)の決め方:
      - yaml由来の ndim が分かっていれば step=ndim を確定的に使う(最優先・曖昧さ無し)。
      - ndim 未知で n_kpt のみ分かれば step = len(rest)//n_kpt。
      - どちらも無いときだけ len(rest)%3 のヒューリスティックにフォールバックする。
        (この経路は K が3の倍数 かつ v無し のとき誤判定しうるので最後の手段)
    """
    insts = []
    if not Path(label_path).exists():
        return insts
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        cx, cy, w, h = map(float, vals[1:5])
        bx1 = (cx - w / 2) * img_w
        by1 = (cy - h / 2) * img_h
        bx2 = (cx + w / 2) * img_w
        by2 = (cy + h / 2) * img_h
        rest = list(map(float, vals[5:]))
        if not rest:
            step, has_v = 3, False
        elif ndim in (2, 3):                       # yaml由来 ndim を最優先
            step, has_v = ndim, (ndim == 3)
        elif n_kpt:                                 # 点数だけ分かる場合
            step = max(2, len(rest) // n_kpt)
            has_v = (step == 3)
        elif len(rest) % 3 == 0:                    # 最後の手段(曖昧)
            step, has_v = 3, True
        else:
            step, has_v = 2, False
        kpts, vis = [], []
        for i in range(0, len(rest) - step + 1, step):
            kpts.append((rest[i] * img_w, rest[i + 1] * img_h))
            vis.append(rest[i + 2] if has_v else 2.0)
        insts.append({
            "bbox": (bx1, by1, bx2, by2),
            "kpts": np.array(kpts, dtype=np.float32).reshape(-1, 2),
            "vis": np.array(vis, dtype=np.float32),
        })
    return insts


def match_by_bbox_center(gt_insts, pred_boxes_xyxy, max_dist_frac=0.5):
    """
    GTとpredをbbox中心の最近傍で1対1対応づける(L字頂点は対象数が少ない前提の素朴マッチ)。
    距離ゲート付き: GTのbbox対角の max_dist_frac 倍を超える対応は『誤検出/検出漏れ』として棄却する。
    これにより無関係な検出(別物体)を頂点誤差に混ぜず、検出漏れとして正しく扱える。
    戻り値: [(gt_index, pred_index), ...] (距離ゲート内で対応がついた組のみ)
    """
    if len(gt_insts) == 0 or len(pred_boxes_xyxy) == 0:
        return []
    gt_boxes = [g["bbox"] for g in gt_insts]
    gt_c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in gt_boxes])
    pr_c = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in pred_boxes_xyxy])
    pairs, used = [], set()
    for gi in range(len(gt_c)):
        bx1, by1, bx2, by2 = gt_boxes[gi]
        diag = float(np.hypot(bx2 - bx1, by2 - by1))
        gate = max(diag * max_dist_frac, 1.0)   # 最低1pxは許容
        d = np.linalg.norm(pr_c - gt_c[gi], axis=1)
        order = np.argsort(d)
        for pi in order:
            pi = int(pi)
            if pi in used:
                continue
            if d[pi] > gate:        # 最近傍でもゲート外なら検出漏れ扱い(以降も全て遠い)
                break
            pairs.append((gi, pi))
            used.add(pi)
            break
    return pairs


# ------------------------------------------------------------------ #
# 精度計測: pred keypoint vs GT keypoint の pixel距離
# ------------------------------------------------------------------ #

def run_accuracy(args):
    """val画像でkeypointのpixel誤差(平均/中央値/最大)を計測する"""
    n_kpt = ndim = None
    if args.data:
        img_dir, lbl_dir, n_kpt, ndim = load_yaml_paths(args.data)
    else:
        img_dir, lbl_dir = Path(args.images), Path(args.labels)
    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)

    model = YOLO(args.model)

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not imgs:
        print(f"[ERR] 画像が見つかりません: {img_dir}")
        return

    all_dist = []          # 全keypointのpixel誤差
    per_kpt = {}           # 頂点index別の誤差リスト
    n_imgs = n_gt = n_pred = n_matched = 0

    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        n_imgs += 1

        gt = parse_gt_label(lbl_dir / (img_path.stem + ".txt"), w, h,
                            n_kpt=n_kpt, ndim=ndim)
        n_gt += len(gt)

        res = model(frame, conf=args.conf, verbose=False)[0]
        if res.keypoints is None or len(res.boxes) == 0:
            continue
        pred_kpts = res.keypoints.xy.cpu().numpy()      # (N, K, 2) 元画像pixel
        pred_boxes = res.boxes.xyxy.cpu().numpy()        # (N, 4)
        n_pred += len(pred_boxes)

        for gi, pi in match_by_bbox_center(gt, pred_boxes):
            gk = gt[gi]["kpts"]
            gv = gt[gi]["vis"]
            pk = pred_kpts[pi]
            k = min(len(gk), len(pk))
            n_matched += 1
            for j in range(k):
                if gv[j] < 0.5:           # 不可視点は誤差計測から除外
                    continue
                d = float(np.linalg.norm(pk[j] - gk[j]))
                all_dist.append(d)
                per_kpt.setdefault(j, []).append(d)

    _print_accuracy_report(all_dist, per_kpt, n_imgs, n_gt, n_pred, n_matched)


def _print_accuracy_report(all_dist, per_kpt, n_imgs, n_gt, n_pred, n_matched):
    """精度レポートを標準出力に整形表示する"""
    print("\n===== Keypoint 精度レポート (pixel誤差) =====")
    print(f"画像枚数         : {n_imgs}")
    print(f"GTインスタンス数 : {n_gt}")
    print(f"検出インスタンス : {n_pred}")
    print(f"マッチ成立数     : {n_matched}")
    if not all_dist:
        print("[WARN] 有効なkeypoint対応が0件でした(検出失敗 or ラベル不一致)")
        return
    arr = np.array(all_dist)
    print("\n--- 全頂点まとめ ---")
    print(f"平均誤差   mean   : {arr.mean():7.2f} px")
    print(f"中央値     median : {np.median(arr):7.2f} px")
    print(f"最大誤差   max    : {arr.max():7.2f} px")
    print(f"95%ile            : {np.percentile(arr, 95):7.2f} px")
    print(f"標準偏差   std    : {arr.std():7.2f} px")
    print(f"対象点数          : {len(arr)}")

    print("\n--- 頂点index別 平均/中央/最大 (px) ---")
    print(f"{'kpt':>4} {'n':>5} {'mean':>8} {'median':>8} {'max':>8}")
    for j in sorted(per_kpt):
        v = np.array(per_kpt[j])
        print(f"{j:>4} {len(v):>5} {v.mean():>8.2f} {np.median(v):>8.2f} {v.max():>8.2f}")


# ------------------------------------------------------------------ #
# 安定性計測: フレーム間ジッター(std) と 検出継続率
# ------------------------------------------------------------------ #

def run_stability(args):
    """動画/連番フレームでkeypoint座標のフレーム間標準偏差と検出継続率を計測する"""
    model = YOLO(args.model)

    # フレーム供給源: 動画 or 画像フォルダ(連番)
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"[ERR] 動画を開けません: {args.video}")
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_iter = _video_frames(cap, args.max_frames)
        src_desc = f"video={args.video} (total={total})"
    else:
        fdir = Path(args.frames)
        files = sorted([p for p in fdir.iterdir() if p.suffix.lower() in IMG_EXTS])
        files = files[: args.max_frames] if args.max_frames else files
        frame_iter = ((cv2.imread(str(p)), p.name) for p in files)
        src_desc = f"frames={fdir} ({len(files)}枚)"

    # トラッキングで同一対象に揃える(persist=Trueで前フレームのIDを引き継ぐ)
    # track_id -> {kpt_index -> [(x,y), ...]} で時系列を蓄積
    series = {}                 # tid -> list of (K,2) ndarray
    track_seen = {}             # tid -> 出現フレーム数
    n_frames = n_detected = 0

    for frame, tag in frame_iter:
        if frame is None:
            continue
        n_frames += 1
        res = model.track(frame, persist=True, conf=args.conf,
                          tracker="bytetrack.yaml", verbose=False)[0]

        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            continue
        ids = res.boxes.id
        if ids is None:
            # IDが付かない場合は単一対象とみなし固定IDで扱う
            ids = np.zeros(len(res.boxes), dtype=int)
        else:
            ids = ids.cpu().numpy().astype(int)
        kpts = res.keypoints.xy.cpu().numpy()       # (N, K, 2) 元画像pixel

        n_detected += 1
        for n, tid in enumerate(ids):
            series.setdefault(tid, []).append(kpts[n])
            track_seen[tid] = track_seen.get(tid, 0) + 1

    if args.video:
        cap.release()

    _print_stability_report(series, track_seen, n_frames, n_detected, src_desc)


def _video_frames(cap, max_frames):
    """動画から連続フレームを取り出すジェネレータ"""
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame, f"f{i:05d}"
        i += 1
        if max_frames and i >= max_frames:
            break


def _print_stability_report(series, track_seen, n_frames, n_detected, src_desc):
    """安定性(ジッター・継続率)レポートを表示する"""
    print("\n===== Keypoint 安定性レポート (ジッター/継続率) =====")
    print(f"ソース           : {src_desc}")
    print(f"総フレーム数     : {n_frames}")
    det_rate = (n_detected / n_frames * 100) if n_frames else 0.0
    print(f"検出フレーム数   : {n_detected}  (検出継続率 {det_rate:5.1f}%)")
    print(f"追跡ID数         : {len(series)}")

    if not series:
        print("[WARN] 追跡対象が得られませんでした")
        return

    # 出現フレーム数が最大のトラックを「主対象」として詳細を出す
    main_tid = max(track_seen, key=track_seen.get)
    print(f"\n--- 主対象 track_id={main_tid} の詳細 ---")
    print(f"出現フレーム数   : {track_seen[main_tid]} "
          f"(対象継続率 {track_seen[main_tid] / n_frames * 100:5.1f}%)")

    seq = np.array(series[main_tid])    # (T, K, 2)
    T, K, _ = seq.shape

    # 各頂点ごとに x,y の標準偏差(=ジッター)。小さいほどライブで安定。
    print(f"\n{'kpt':>4} {'std_x':>8} {'std_y':>8} {'std_2d':>8} {'jump_max':>9}")
    overall = []
    for j in range(K):
        xs, ys = seq[:, j, 0], seq[:, j, 1]
        sx, sy = float(xs.std()), float(ys.std())
        s2d = float(np.sqrt(sx ** 2 + sy ** 2))   # 2D合成ジッター
        # フレーム間の移動量(連続差分)の最大 = 瞬間的な飛び(スパイク)
        diffs = np.linalg.norm(np.diff(seq[:, j, :], axis=0), axis=1) if T > 1 else np.array([0.0])
        jump_max = float(diffs.max()) if diffs.size else 0.0
        overall.append(s2d)
        print(f"{j:>4} {sx:>8.2f} {sy:>8.2f} {s2d:>8.2f} {jump_max:>9.2f}")

    print(f"\n全頂点 平均2Dジッター : {statistics.mean(overall):6.2f} px")
    print(f"全頂点 最大2Dジッター : {max(overall):6.2f} px")
    print("(目安: std_2d が小さいほどライブ映像で頂点がブレない=安定)")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def build_parser():
    """サブコマンド(accuracy/stability)のCLIを構築する"""
    p = argparse.ArgumentParser(description="YOLO Pose Keypoint 精度・安定性 計測")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("accuracy", help="val画像でpixel誤差を計測")
    pa.add_argument("--model", required=True, help="best.pt のパス")
    pa.add_argument("--data", help="dataset.yaml (val/labelsを自動推定)")
    pa.add_argument("--images", help="val画像ディレクトリ(--data未指定時)")
    pa.add_argument("--labels", help="valラベルディレクトリ(--data未指定時)")
    pa.add_argument("--conf", type=float, default=0.25, help="検出confしきい値")
    pa.set_defaults(func=run_accuracy)

    ps = sub.add_parser("stability", help="動画/連番でジッター・継続率を計測")
    ps.add_argument("--model", required=True, help="best.pt のパス")
    ps.add_argument("--video", help="動画ファイル")
    ps.add_argument("--frames", help="連番フレームのディレクトリ(--video未指定時)")
    ps.add_argument("--conf", type=float, default=0.10, help="検出confしきい値(安定性は低めで継続率を見る)")
    ps.add_argument("--max-frames", type=int, default=0, help="先頭Nフレームのみ(0=全部)")
    ps.set_defaults(func=run_stability)
    return p


def main():
    """引数を解釈してサブコマンドを実行する"""
    args = build_parser().parse_args()
    if args.cmd == "accuracy" and not (args.data or (args.images and args.labels)):
        print("[ERR] accuracy には --data か (--images と --labels) が必要です")
        return
    if args.cmd == "stability" and not (args.video or args.frames):
        print("[ERR] stability には --video か --frames が必要です")
        return
    args.func(args)


if __name__ == "__main__":
    main()
