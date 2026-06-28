# 作成: 2026-06-28
# 目的: Make Senseのような社内アノテーションツール。
#       UVCライブ映像/静止画から矩形を描き、YOLOフォーマット(.txt)で保存する。

import os
import sys
import glob
import json

import cv2
import numpy as np
from onnx_detector import OnnxDetector
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QSize, QProcess
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QHBoxLayout, QVBoxLayout,
    QSpinBox, QMessageBox, QTabWidget, QStackedWidget, QListView,
    QButtonGroup, QGraphicsDropShadowEffect, QDialog, QPlainTextEdit,
    QDialogButtonBox, QLineEdit,
)

# キャプチャ画像の保存先（exe/スクリプトと同じ場所。書き込み不可なら一時フォルダ）
def _capture_dir():
    """書き込み可能なキャプチャ保存先を決める"""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base, "captures")
    try:
        os.makedirs(candidate, exist_ok=True)
        test = os.path.join(candidate, ".write_test")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return candidate
    except Exception:
        # 書き込み不可ならユーザーのTempフォルダにフォールバック
        import tempfile
        fallback = os.path.join(tempfile.gettempdir(), "pyYOLO_captures")
        os.makedirs(fallback, exist_ok=True)
        return fallback

CAPTURE_DIR = _capture_dir()

# クラス定義の保存先（exe/スクリプトと同じ場所のclasses.json）
def _base_dir():
    """exe/スクリプトの配置ディレクトリを返す"""
    return os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))

CONFIG_PATH = os.path.join(_base_dir(), "classes.json")
DEFAULT_CLASSES = ["l_mark", "cross", "circle_cross", "l_mark_black"]

def load_classes():
    """classes.jsonからクラス名一覧を読み込む（無ければ既定値）"""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return [str(x) for x in data]
    except Exception:
        pass
    return list(DEFAULT_CLASSES)

def save_classes(classes):
    """クラス名一覧をclasses.jsonに保存する"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

# 対象クラス（実行時に編集可能。class_id = リストの並び順）
CLASSES = load_classes()

# --- YOLO Pose(Keypoint)設定 ---
# kpt_shape = [キーポイント数K, 次元D]。本プロジェクトは L字頂点1点・(x,y,visible) の D=3。
# pose形式の1行 = "cls cx cy w h  px1 py1 v1 ... pxK pyK vK"（全て0..1正規化）。
# 列数は固定: 5 + K*D。混在(5値行と8値行)はultralytics Poseローダが弾くため、
# pose保存時は頂点未指定の矩形も px=py=0, v=0 で必ず全列書き出す(v=0は損失から除外される)。
KPT_COUNT = 1   # K: キーポイント数
KPT_DIMS = 3    # D: x,y,visible
POSE_NCOLS = 5 + KPT_COUNT * KPT_DIMS   # pose 1行の列数(K=1,D=3 → 8)

# クラス描画色のパレット（クラス数より多めに用意し、足りなければ循環）
PALETTE = [
    QColor(0, 255, 0), QColor(0, 180, 255), QColor(255, 180, 0), QColor(255, 0, 180),
    QColor(255, 255, 0), QColor(0, 255, 255), QColor(255, 80, 80), QColor(160, 120, 255),
]

def class_color(i):
    """クラスIDに対応する描画色を返す（パレットを循環）"""
    return PALETTE[i % len(PALETTE)]

def class_name(i):
    """クラスIDに対応する名前を返す（範囲外は仮名）"""
    return CLASSES[i] if 0 <= i < len(CLASSES) else f"id{i}"


class Canvas(QWidget):
    """画像表示と矩形描画を担当するキャンバスウィジェット"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.image = None            # 元画像（BGR np.ndarray）
        self.boxes = []              # [(cls_id, x1, y1, x2, y2)] 元画像ピクセル座標
        self.current_class = 0       # 現在選択中のクラスID
        self.drawing = False         # ドラッグ中フラグ
        self.start_pt = QPoint()     # ドラッグ開始点（ウィジェット座標）
        self.cur_pt = QPoint()       # ドラッグ中の現在点（ウィジェット座標）
        self._scale = 1.0            # 表示倍率（fit倍率×ズーム）
        self._off_x = 0.0            # 表示時の左オフセット（float）
        self._off_y = 0.0            # 表示時の上オフセット（float）
        self.live = False            # カメラ起動中フラグ（緑グロー枠の表示用）
        # 表示モード: タブごとに「何を描くか」を決める（live機能フラグとは独立）
        #   "annotate"=アノテ系一式 / "camera"=ライブ+グロー /
        #   "inference"=ライブ+検出枠+グロー / "blank"=何も描かない
        self.display_mode = "annotate"
        self.overlay_text = ""       # 画像右上に出す番号オーバーレイ（例 "13/20"）
        self.detections = []         # 推論結果の表示用 [dict(x1,y1,x2,y2,conf,cls)]
        # --- キーポイントモード（YOLO Pose用：L字頂点を矩形ごとに1点指定）---
        self.kpt_mode = False        # キーポイント指定モードON/OFF（Kキーで切替）
        self.keypoints = []          # boxesと並走。各要素は (px, py) 元画像px座標 or None（未指定）
        # --- ズーム/パン（マウススクロールで拡大・精密な頂点指定用）---
        self.zoom = 1.0              # 拡大率（1.0=全体表示〜12.0）
        self._pan_x = 0.0            # ズーム時の追加オフセットX（パン）
        self._pan_y = 0.0            # ズーム時の追加オフセットY（パン）
        self._panning = False        # 中ボタンドラッグでパン中
        self._pan_start = QPoint()   # パン開始時のマウス位置
        self.hover_pt = None         # キーポイントモードのガイド線用カーソル位置
        self.guide_angle = 0.0       # ガイド十字線の回転角（度）。Q=CCW / R=CW
        self.setMouseTracking(True)  # ボタン無しでもマウス移動を受ける（ガイド線用）

    def set_live(self, flag):
        """カメラ起動中フラグを切り替えて再描画する"""
        self.live = flag
        self.update()

    def set_image(self, bgr):
        """新しい画像をセットして再描画する"""
        self.image = bgr
        self.boxes = []
        self.keypoints = []          # キーポイントも初期化
        self.detections = []
        self.reset_view()            # ズーム/パンを初期化
        self.update()

    def reset_view(self):
        """ズーム・パンを初期状態（全体表示）に戻す"""
        self.zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.hover_pt = None

    def set_class(self, cls_id):
        """描画対象のクラスIDを切り替える"""
        self.current_class = cls_id

    def clear_boxes(self):
        """全ての矩形を削除する"""
        self.boxes = []
        self.keypoints = []          # 対応するキーポイントも全消去
        self.update()

    def delete_box(self, index):
        """指定インデックスの矩形を削除する（対応するキーポイントも消す）"""
        if 0 <= index < len(self.boxes):
            del self.boxes[index]
            if index < len(self.keypoints):
                del self.keypoints[index]
            self.update()

    def set_kpt_mode(self, flag):
        """キーポイント指定モードのON/OFFを切り替えて再描画する"""
        self.kpt_mode = flag
        self.update()

    def rotate_guide(self, deg):
        """ガイド十字線を回転する（キーポイントモード時のみ）"""
        if not self.kpt_mode:
            return
        self.guide_angle = (self.guide_angle + deg) % 360.0
        self.update()

    def _box_at(self, ix, iy):
        """元画像座標(ix,iy)を含む矩形のindexを返す（複数なら最小面積を優先・無ければNone）"""
        best, best_area = None, None
        for i, (cls_id, x1, y1, x2, y2) in enumerate(self.boxes):
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best, best_area = i, area
        return best

    def _fit_scale(self):
        """画像をウィジェットに収めるfit倍率（ズーム1.0時の倍率）"""
        h, w = self.image.shape[:2]
        return min(self.width() / w, self.height() / h)

    def _calc_transform(self):
        """fit倍率×ズーム＋パンから表示倍率とオフセットを計算する"""
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        self._scale = self._fit_scale() * self.zoom
        disp_w, disp_h = w * self._scale, h * self._scale
        self._off_x = (self.width() - disp_w) / 2 + self._pan_x
        self._off_y = (self.height() - disp_h) / 2 + self._pan_y

    def _clamp_pan(self):
        """拡大時に画像が枠から離れすぎないようパンを制限する"""
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        scale = self._fit_scale() * self.zoom
        disp_w, disp_h = w * scale, h * scale
        base_x = (self.width() - disp_w) / 2
        base_y = (self.height() - disp_h) / 2
        # 画像が枠より大きい軸はオフセットを [枠-表示, 0] に収める。小さい軸は中央固定
        if disp_w >= self.width():
            self._pan_x = min(max(self._pan_x, (self.width() - disp_w) - base_x), -base_x)
        else:
            self._pan_x = 0.0
        if disp_h >= self.height():
            self._pan_y = min(max(self._pan_y, (self.height() - disp_h) - base_y), -base_y)
        else:
            self._pan_y = 0.0

    def _img_to_widget(self, ix, iy):
        """元画像座標をウィジェット座標(int)に変換する"""
        return int(ix * self._scale + self._off_x), int(iy * self._scale + self._off_y)

    def wheelEvent(self, event):
        """マウスホイールでカーソル位置を中心に拡大/縮小する"""
        if self.image is None:
            return
        self._calc_transform()
        cx, cy = event.position().x(), event.position().y()
        # ズーム前のカーソル直下の元画像座標
        ix = (cx - self._off_x) / self._scale
        iy = (cy - self._off_y) / self._scale
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        new_zoom = max(1.0, min(12.0, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        h, w = self.image.shape[:2]
        new_scale = self._fit_scale() * self.zoom
        # カーソル直下の点が同じ位置に留まるようパンを決める
        self._pan_x = cx - ix * new_scale - (self.width() - w * new_scale) / 2
        self._pan_y = cy - iy * new_scale - (self.height() - h * new_scale) / 2
        if self.zoom == 1.0:
            self._pan_x = self._pan_y = 0.0
        self._clamp_pan()
        self.update()

    def _widget_to_image(self, pt):
        """ウィジェット座標を元画像ピクセル座標に変換する"""
        x = (pt.x() - self._off_x) / self._scale
        y = (pt.y() - self._off_y) / self._scale
        h, w = self.image.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return int(x), int(y)

    def paintEvent(self, event):
        """表示モードに応じて画像・オーバーレイ・検出枠を描き分ける"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        mode = getattr(self, "display_mode", "annotate")

        # blank（学習・変換タブ）は灰背景＋案内のみ。画像もオーバーレイも描かない
        if mode == "blank":
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "学習・変換タブ（プレビューなし）")
            return

        # camera は live時のみライブ映像。未起動は残像を見せず案内文
        if mode == "camera" and not self.live:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "カメラ開始を押してください")
            return
        # inference は カメラ推論(live) か 静止画推論(detections有) のとき描画
        if mode == "inference" and not self.live and not self.detections:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "モデル読込後にカメラで推論 or 表示中画像で推論")
            return

        # annotate で画像が無いときは案内文
        if self.image is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "画像またはカメラを読み込んでください")
            return

        self._calc_transform()
        # BGR→RGBに変換してQImageを作る
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        target = QRect(int(self._off_x), int(self._off_y), int(w * self._scale), int(h * self._scale))
        painter.drawImage(target, qimg)

        is_annotate = (mode == "annotate")

        # --- アノテーション系（矩形・頂点・ドラッグ枠）は annotate のみ ---
        if is_annotate:
            for cls_id, x1, y1, x2, y2 in self.boxes:
                self._draw_box(painter, cls_id, x1, y1, x2, y2)
            for i, kpt in enumerate(self.keypoints):
                if kpt is None:
                    continue
                cls_id = self.boxes[i][0] if i < len(self.boxes) else 0
                self._draw_keypoint(painter, kpt[0], kpt[1], cls_id)
            if self.drawing:
                painter.setPen(QPen(class_color(self.current_class), 2, Qt.DashLine))
                painter.drawRect(QRect(self.start_pt, self.cur_pt))

        # --- 緑グロー：camera / inference（live時のみ）---
        if mode in ("camera", "inference") and self.live:
            for width, alpha in [(14, 35), (10, 70), (6, 130), (2, 230)]:
                painter.setPen(QPen(QColor(0, 255, 0, alpha), width))
                m = width // 2
                painter.drawRect(self.rect().adjusted(m, m, -m, -m))

        # --- 検出枠（黄）：inference のみ ---
        if mode == "inference" and self.detections:
            painter.setFont(QFont("Meiryo", 10, QFont.Bold))
            for d in self.detections:
                wx1, wy1 = self._img_to_widget(d["x1"], d["y1"])
                wx2, wy2 = self._img_to_widget(d["x2"], d["y2"])
                painter.setPen(QPen(QColor(255, 230, 0), 2))
                painter.drawRect(QRect(QPoint(wx1, wy1), QPoint(wx2, wy2)))
                painter.drawText(wx1, wy1 - 4, f"{class_name(d['cls'])} {d['conf']:.2f}")

        # 以降のガイド線・番号・KPTラベルは annotate 専用
        if not is_annotate:
            return

        # キーポイントモード中はカーソル位置に回転可能なガイド十字線を引く
        if self.kpt_mode and self.hover_pt is not None:
            disp_w, disp_h = w * self._scale, h * self._scale
            img_rect = QRect(int(self._off_x), int(self._off_y), int(disp_w), int(disp_h))
            hx, hy = self.hover_pt.x(), self.hover_pt.y()
            if img_rect.contains(self.hover_pt):
                painter.save()
                painter.setClipRect(img_rect)   # 画像範囲外へはみ出さない
                length = (self.width() ** 2 + self.height() ** 2) ** 0.5
                a = np.radians(self.guide_angle)
                painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
                for ang in (a, a + np.pi / 2):   # 直交する2本（L字の2辺に対応）
                    dx, dy = np.cos(ang) * length, np.sin(ang) * length
                    painter.drawLine(int(hx - dx), int(hy - dy), int(hx + dx), int(hy + dy))
                painter.restore()

        # 単一画面の右上に画像番号をオーバーレイ表示する
        if self.overlay_text:
            painter.setFont(QFont("Meiryo", 14, QFont.Bold))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(self.overlay_text)
            th = fm.height()
            pad = 6
            disp_w = self.image.shape[1] * self._scale
            x = int(self._off_x + disp_w - tw - pad * 2 - 8)
            y = int(self._off_y + 8)
            painter.fillRect(x, y, tw + pad * 2, th + pad, QColor(0, 0, 0, 160))
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(x + pad, y + th - 2, self.overlay_text)

        # キーポイントモード中は左上に小さく表示する
        if self.kpt_mode:
            painter.setFont(QFont("Meiryo", 11, QFont.Bold))
            painter.setPen(QColor(255, 0, 255))
            label = "KEYPOINT MODE (K)"
            if self.zoom > 1.01:
                label += f"  x{self.zoom:.1f}"
            label += f"  ガイド{self.guide_angle:.1f}° (Q/E, Shiftで0.5°)"
            painter.drawText(int(self._off_x) + 8, int(self._off_y) + 22, label)

    def _draw_keypoint(self, painter, px, py, cls_id):
        """キーポイント（頂点）を小さな十字＋中心点で描画する"""
        wx, wy = self._img_to_widget(px, py)
        r = 7  # 十字の腕の長さ（ウィジェットpx）
        # 視認性のため白の縁取り→マゼンタ本体の二重描き
        for color, width in [(QColor(255, 255, 255), 4), (QColor(255, 0, 255), 2)]:
            painter.setPen(QPen(color, width))
            painter.drawLine(wx - r, wy, wx + r, wy)
            painter.drawLine(wx, wy - r, wx, wy + r)

    def _draw_box(self, painter, cls_id, x1, y1, x2, y2):
        """1つの矩形とクラス名ラベルを描画する"""
        color = class_color(cls_id)
        wx1, wy1 = self._img_to_widget(x1, y1)
        wx2, wy2 = self._img_to_widget(x2, y2)
        painter.setPen(QPen(color, 2))
        painter.drawRect(QRect(QPoint(wx1, wy1), QPoint(wx2, wy2)))
        painter.setFont(QFont("Meiryo", 9))
        painter.drawText(wx1, wy1 - 4, class_name(cls_id))

    def mousePressEvent(self, event):
        """ドラッグ開始：矩形の始点を記録する（キーポイントモードでは頂点を1点指定）"""
        if self.image is None:
            return
        # 中ボタンドラッグでパン（拡大時の移動）
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            return
        if event.button() != Qt.LeftButton:
            return
        # キーポイントモード：クリック点を含む矩形を探し、その頂点を登録する
        if self.kpt_mode:
            ix, iy = self._widget_to_image(event.position().toPoint())
            idx = self._box_at(ix, iy)
            win = self.window()
            if idx is None:
                if hasattr(win, "set_status"):
                    win.set_status("キーポイントは矩形の内側をクリックして指定してください")
                return
            # boxesとkeypointsの長さを揃えてから該当indexに格納
            while len(self.keypoints) < len(self.boxes):
                self.keypoints.append(None)
            self.keypoints[idx] = (ix, iy)
            self.update()
            if hasattr(win, "refresh_list"):
                win.refresh_list()
            return
        # 通常モード：矩形ドラッグ開始
        self.drawing = True
        self.start_pt = event.position().toPoint()
        self.cur_pt = self.start_pt

    def mouseMoveEvent(self, event):
        """ドラッグ中の矩形更新／パン／ガイド線用のカーソル追従"""
        pos = event.position().toPoint()
        self.hover_pt = pos          # ガイド線用に常に保持
        if self._panning:
            # 中ボタンドラッグ量だけパンを動かす
            self._pan_x += pos.x() - self._pan_start.x()
            self._pan_y += pos.y() - self._pan_start.y()
            self._pan_start = pos
            self._clamp_pan()
            self.update()
            return
        if self.drawing:
            self.cur_pt = pos
            self.update()
        elif self.kpt_mode:
            self.update()            # キーポイントモードはガイド線を毎フレーム再描画

    def mouseReleaseEvent(self, event):
        """ドラッグ終了：矩形を確定して保存する"""
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            return
        if not self.drawing:
            return
        self.drawing = False
        x1, y1 = self._widget_to_image(self.start_pt)
        x2, y2 = self._widget_to_image(event.position().toPoint())
        # 小さすぎる矩形は無視する
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            self.update()
            return
        box = (self.current_class, min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.boxes.append(box)
        self.update()
        # 親ウィンドウのリストを更新する
        win = self.window()
        if hasattr(win, "refresh_list"):
            win.refresh_list()


class ClassSettingsDialog(QDialog):
    """クラス名を1行1クラスで編集する設定ダイアログ"""

    def __init__(self, classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("クラス設定")
        self.resize(360, 320)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("1行に1クラス（上から class_id 0,1,2…）\n※ 注釈後に順番/個数を変えるとIDがずれます"))
        self.editor = QPlainTextEdit("\n".join(classes))
        lay.addWidget(self.editor)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def get_classes(self):
        """入力テキストから空行を除いたクラス名一覧を返す"""
        lines = [ln.strip() for ln in self.editor.toPlainText().splitlines()]
        return [ln for ln in lines if ln]


class MainWindow(QMainWindow):
    """アノテーションツールのメインウィンドウ"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("pyYOLO アノテーションツール")
        self.resize(1100, 720)
        self.cap = None              # cv2.VideoCapture（カメラ）
        self.timer = QTimer()        # ライブ映像更新用タイマー
        self.timer.timeout.connect(self._update_camera)
        self.detector = None         # OnnxDetector（推論モデル）
        self.infer_timer = QTimer()  # 推論用タイマー
        self.infer_timer.timeout.connect(self._infer_frame)
        self.image_files = []        # フォルダ読込時の画像パス一覧
        self.cur_index = -1          # 現在表示中の画像インデックス
        self.cur_path = None         # 現在表示中の画像パス
        self._build_ui()

    def _build_ui(self):
        """ウィジェットを配置してUIを組み立てる（左パネルはタブ構成）"""
        self.canvas = Canvas()

        # タブで「カメラ撮影」と「アノテーション」を分ける
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_camera_tab(), "カメラ撮影")
        self.tabs.addTab(self._build_annotation_tab(), "アノテーション")
        self.tabs.addTab(self._build_train_tab(), "学習・変換")
        self.tabs.addTab(self._build_inference_tab(), "推論")
        self.tabs.currentChanged.connect(self._on_tab_changed)  # タブ切替で案内を更新

        left = QVBoxLayout()
        left.addWidget(self.tabs)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(280)

        # 右側：単一画面（canvas）と複数画面（gallery）をスタックで切替える
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.IconMode)
        self.gallery.setIconSize(QSize(260, 195))  # サムネイルを大きめに
        self.gallery.setResizeMode(QListView.Adjust)
        self.gallery.setMovement(QListView.Static)
        self.gallery.setSpacing(10)
        self.gallery.itemClicked.connect(self._on_gallery_click)

        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.canvas)   # index 0 = 単一画面
        self.view_stack.addWidget(self.gallery)  # index 1 = 複数画面

        # 右側上部：クラス選択のトグルボタン列 + その下に表示スタック
        self.class_bar = QHBoxLayout()
        self.class_bar.setContentsMargins(4, 4, 4, 4)
        self.class_bar.setSpacing(6)
        class_bar_widget = QWidget()
        class_bar_widget.setLayout(self.class_bar)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(class_bar_widget)
        right.addWidget(self.view_stack, stretch=1)
        right_widget = QWidget()
        right_widget.setLayout(right)

        root = QHBoxLayout()
        root.addWidget(left_widget)
        root.addWidget(right_widget, stretch=1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        # メニュー：設定 > クラス設定
        menu = self.menuBar().addMenu("設定")
        act = menu.addAction("クラス設定…")
        act.triggered.connect(self.open_class_settings)

        # クラスボタンを構築（F1〜のショートカット付き）
        self.class_buttons = []
        self.class_group = QButtonGroup(self)
        self.class_group.setExclusive(True)
        self._rebuild_class_bar()

        # 画面下部のステータスバー（QLabelを埋め込み左端の見切れを防ぐ）
        self.status_label = QLabel("準備完了")
        self.status_label.setContentsMargins(8, 0, 8, 0)
        self.statusBar().addWidget(self.status_label, 1)
        self._on_tab_changed(0)

    def _rebuild_class_bar(self):
        """クラス選択ボタン列を作り直す（クラス設定変更時にも呼ぶ）"""
        # 既存ボタンを除去
        for b in self.class_buttons:
            self.class_group.removeButton(b)
            b.setParent(None)
        self.class_buttons = []
        # クラスごとにトグルボタンを生成（F1〜F12対応）
        for i, name in enumerate(CLASSES):
            label = f"F{i + 1}: {name}" if i < 12 else name
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda _=False, idx=i: self.select_class(idx))
            self.class_group.addButton(btn, i)
            self.class_bar.addWidget(btn)
            self.class_buttons.append(btn)
        self.class_bar.addStretch(1)
        # 現在クラスを範囲内に収めて選択
        cur = min(self.canvas.current_class, len(CLASSES) - 1) if CLASSES else 0
        if self.class_buttons:
            self.select_class(cur)

    def select_class(self, idx):
        """クラスを選択し、アクティブボタンを緑グローで強調する"""
        if not (0 <= idx < len(self.class_buttons)):
            return
        self.canvas.set_class(idx)
        for i, b in enumerate(self.class_buttons):
            active = (i == idx)
            b.setChecked(active)
            if active:
                # アクティブは枠線なし・テキストのみ蛍光緑グロー
                glow = QGraphicsDropShadowEffect(b)
                glow.setBlurRadius(18)
                glow.setColor(QColor(57, 255, 20))
                glow.setOffset(0, 0)
                b.setGraphicsEffect(glow)
                b.setStyleSheet("QPushButton{color:#39ff14; font-weight:bold;}")
            else:
                b.setGraphicsEffect(None)
                b.setStyleSheet("")

    def open_class_settings(self):
        """クラス設定ダイアログを開いて名前一覧を編集する"""
        dlg = ClassSettingsDialog(CLASSES, self)
        if dlg.exec() == QDialog.Accepted:
            new_classes = dlg.get_classes()
            if not new_classes:
                QMessageBox.warning(self, "警告", "クラスを1つ以上指定してください")
                return
            save_classes(new_classes)  # ファイルにのみ保存（再起動後に反映）
            self.set_status(f"クラスを保存しました（{len(new_classes)}件）。再起動後に有効になります")
            QMessageBox.information(self, "クラス設定",
                                    "クラス設定を保存しました。\nアプリを再起動すると反映されます。")

    def set_status(self, msg):
        """ステータスバーにメッセージを表示する"""
        self.status_label.setText(msg)

    def _on_tab_changed(self, index):
        """タブ切替時に表示モードと stale 状態を設定し、ガイド/ボタンを更新する"""
        # タブを離れたらライブ（カメラ/推論）を止める（残像・検出枠の残りを防ぐ）
        if index != 0:
            self._stop_camera()
        if index != 3:
            self._stop_inference()

        # 表示モードをタブに対応づける（paintはこのモードで分岐する）
        self.canvas.display_mode = {0: "camera", 1: "annotate", 2: "blank", 3: "inference"}.get(index, "blank")

        # アノテタブ以外では右側を単一画面に固定（gallery が裏に残るのを防ぐ）
        if index != 1:
            self.view_stack.setCurrentIndex(0)
        # 推論タブ以外では検出枠を内部的にもクリア
        if index != 3:
            self.canvas.detections = []
        # アノテタブ以外ではキーポイント/操作モードの stale をリセット
        if index != 1:
            self.canvas.kpt_mode = False
            self.canvas.hover_pt = None
            self.canvas.guide_angle = 0.0
            self.canvas.drawing = False

        # アノテーションタブに来たら、ライブ残像でなく現在の静止画を再表示する
        if index == 1:
            if self.image_files and 0 <= self.cur_index < len(self.image_files):
                self._load_image(self.image_files[self.cur_index])
            else:
                # 残フレーム・古い画像が残らないよう案内文へ戻す
                self.canvas.image = None
                self.canvas.boxes = []
                self.canvas.keypoints = []
                self.canvas.overlay_text = ""

        # クラスボタンはアノテーションタブ(index 1)でのみ有効
        if hasattr(self, "class_buttons"):
            for b in self.class_buttons:
                b.setEnabled(index == 1)

        self.canvas.update()  # モード変更を即時反映
        guides = {
            0: "カメラ撮影：カメラ開始 → Spaceで連続撮影 / Ctrl+Cで停止",
            1: "アノテーション：撮影フォルダを開く → 矩形描画 → Sで保存 / A:前 D:次",
            2: "学習・変換：撮影フォルダ→データセット作成→学習→ONNX変換（開発環境のみ）",
            3: "推論：モデル(.onnx)読込 → カメラで推論 or 表示中画像で推論",
        }
        self.set_status(guides.get(index, ""))

    def _build_camera_tab(self):
        """タブ①：カメラ撮影パネルを組み立てる"""
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 10)

        btn_cam = QPushButton("カメラ開始/停止（Ctrl+Cで停止）")
        btn_cam.clicked.connect(self.toggle_camera)
        btn_capture = QPushButton("キャプチャ（Space）")
        btn_capture.clicked.connect(self.capture_frame)
        # Spaceキーがボタンに吸われないようフォーカスを無効化（必ずkeyPressEventへ）
        btn_cam.setFocusPolicy(Qt.NoFocus)
        btn_capture.setFocusPolicy(Qt.NoFocus)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("カメラ番号"))
        lay.addWidget(self.cam_index)
        lay.addWidget(btn_cam)
        lay.addWidget(btn_capture)
        lay.addWidget(QLabel("※ カメラは止まりません。\n   Spaceで連続撮影できます。\n   撮影後はアノテーションタブの\n   「撮影フォルダを開く」へ"))
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_annotation_tab(self):
        """タブ②：アノテーションパネルを組み立てる（クラス選択は上部ボタン列）"""
        btn_open_file = QPushButton("画像を開く")
        btn_open_file.clicked.connect(self.open_file)
        btn_open_folder = QPushButton("フォルダを開く")
        btn_open_folder.clicked.connect(self.open_folder)
        btn_open_captures = QPushButton("撮影フォルダを開く")
        btn_open_captures.clicked.connect(self.open_capture_folder)
        btn_view = QPushButton("表示切替 単一⇄複数（W）")
        btn_view.setFocusPolicy(Qt.NoFocus)
        btn_view.clicked.connect(self.toggle_view)
        btn_prev = QPushButton("← 前の画像（A）")
        btn_prev.clicked.connect(lambda: self.navigate(-1))
        btn_next = QPushButton("次の画像（D）→")
        btn_next.clicked.connect(lambda: self.navigate(1))
        btn_delete = QPushButton("選択した矩形を削除")
        btn_delete.clicked.connect(self.delete_selected)
        btn_clear = QPushButton("全矩形クリア")
        btn_clear.clicked.connect(self.canvas.clear_boxes)
        btn_save = QPushButton("保存（S）")
        btn_save.setFocusPolicy(Qt.NoFocus)
        btn_save.clicked.connect(self.save_label)

        self.box_list = QListWidget()
        self.box_list.setFocusPolicy(Qt.NoFocus)  # A/Dキーがリストに吸われないように

        lay = QVBoxLayout()
        lay.addWidget(QLabel("クラスは画面上部のボタンで選択（F1〜）"))
        lay.addWidget(btn_open_file)
        lay.addWidget(btn_open_folder)
        lay.addWidget(btn_open_captures)
        lay.addWidget(btn_view)
        lay.addWidget(btn_prev)
        lay.addWidget(btn_next)
        lay.addWidget(QLabel("アノテーション一覧"))
        lay.addWidget(self.box_list)
        lay.addWidget(btn_delete)
        lay.addWidget(btn_clear)
        lay.addWidget(btn_save)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_train_tab(self):
        """タブ③：学習・ONNX変換パネル（開発環境=python実行時のみ動作）"""
        self.model_name_edit = QLineEdit("marks")
        self.train_src_label = QLabel(CAPTURE_DIR)
        self.train_src_label.setWordWrap(True)
        self.train_src = CAPTURE_DIR

        btn_src = QPushButton("撮影フォルダを選択")
        btn_src.clicked.connect(self._choose_train_src)
        btn_make = QPushButton("① データセット作成")
        btn_make.clicked.connect(self.make_dataset)
        btn_train = QPushButton("② 学習開始（best.pt作成）")
        btn_train.clicked.connect(self.start_training)
        btn_export = QPushButton("③ ONNX変換（best.pt→onnx）")
        btn_export.clicked.connect(self.start_export)

        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(500)

        self.train_buttons = [btn_make, btn_train, btn_export]
        lay = QVBoxLayout()
        lay.addWidget(QLabel("モデル名（フォルダ名になります）"))
        lay.addWidget(self.model_name_edit)
        lay.addWidget(QLabel("学習元の撮影フォルダ"))
        lay.addWidget(self.train_src_label)
        lay.addWidget(btn_src)
        lay.addWidget(btn_make)
        lay.addWidget(btn_train)
        lay.addWidget(btn_export)
        lay.addWidget(QLabel("ログ"))
        lay.addWidget(self.train_log)

        # 学習・変換に使えるPython（ultralytics導入済み）を探す
        self.trainer_py = self._trainer_python()
        if self.trainer_py is None:
            for b in (btn_train, btn_export):
                b.setEnabled(False)
            note = QLabel("※ 学習・変換にはPython（ultralytics導入済み）が必要です。\n   このPCで見つかりませんでした。")
            note.setStyleSheet("color:#ff8080;")
            note.setWordWrap(True)
            lay.addWidget(note)
        else:
            note = QLabel(f"学習に使うPython: {self.trainer_py}")
            note.setStyleSheet("color:#80c0ff;")
            note.setWordWrap(True)
            lay.addWidget(note)

        w = QWidget()
        w.setLayout(lay)
        return w

    def _trainer_python(self):
        """学習・変換に使えるPython実行ファイルを探す（py実行ならそれ、exeならPATH探索）"""
        if not getattr(sys, "frozen", False):
            return sys.executable  # python起動ならそのまま使える
        import shutil
        for cand in ("python", "py"):
            p = shutil.which(cand)
            if p:
                return p
        return None

    def _choose_train_src(self):
        """学習元の撮影フォルダを選ぶ"""
        folder = QFileDialog.getExistingDirectory(self, "撮影フォルダを選択", CAPTURE_DIR)
        if folder:
            self.train_src = folder
            self.train_src_label.setText(folder)

    def _train_log(self, text):
        """学習ログ欄に追記する"""
        self.train_log.appendPlainText(text.rstrip())

    def make_dataset(self):
        """撮影フォルダの注釈済み画像をtrain/valに分割しdatasets/<名前>を作る"""
        import random
        name = self.model_name_edit.text().strip() or "model"
        base = _base_dir()
        dst = os.path.join(base, "datasets", name)
        # 注釈済み画像を集める
        pairs = []
        for jpg in sorted(glob.glob(os.path.join(self.train_src, "*.jpg"))):
            txt = os.path.splitext(jpg)[0] + ".txt"
            if os.path.exists(txt) and os.path.getsize(txt) > 0:
                pairs.append((jpg, txt))
        if not pairs:
            QMessageBox.warning(self, "警告", "注釈済み画像が見つかりません")
            return
        random.seed(42)
        random.shuffle(pairs)
        split = max(1, int(len(pairs) * 0.8))
        for s in ("train", "val"):
            os.makedirs(os.path.join(dst, "images", s), exist_ok=True)
            os.makedirs(os.path.join(dst, "labels", s), exist_ok=True)
        import shutil
        for i, (jpg, txt) in enumerate(pairs):
            s = "train" if i < split else "val"
            bn = os.path.splitext(os.path.basename(jpg))[0]
            shutil.copy(jpg, os.path.join(dst, "images", s, bn + ".jpg"))
            shutil.copy(txt, os.path.join(dst, "labels", s, bn + ".txt"))
        # pose形式(5+K*D列)のラベルが含まれるか判定する
        # (含まれる場合は yaml に kpt_shape / flip_idx を付けないと Pose 学習できない)
        is_pose = False
        for _, txt in pairs:
            with open(txt) as tf:
                for line in tf:
                    if len(line.split()) == POSE_NCOLS:
                        is_pose = True
                        break
            if is_pose:
                break
        # data.yamlを書き出す（クラスは現在のCLASSES）
        yaml_path = os.path.join(base, "datasets", f"{name}.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {dst}\n")
            f.write("train: images/train\nval: images/val\n")
            if is_pose:
                # 単一非対称頂点(K=1)。flip_idxは長さKで [0]。
                # 注意: K=1の非対称頂点では学習側で fliplr=0.0 を推奨(train_pose.py参照)。
                f.write(f"kpt_shape: [{KPT_COUNT}, {KPT_DIMS}]\n")
                f.write(f"flip_idx: [{', '.join(str(i) for i in range(KPT_COUNT))}]\n")
            f.write(f"nc: {len(CLASSES)}\nnames:\n")
            for i, c in enumerate(CLASSES):
                f.write(f"  {i}: {c}\n")
        self.train_yaml = yaml_path
        self._train_log(f"データセット作成: {dst}（train {split} / val {len(pairs)-split}）")
        self._train_log(f"yaml: {yaml_path}")
        self.set_status(f"データセット作成完了: {name}")

    def start_training(self):
        """データセットでYOLO学習を開始する（別プロセス）"""
        if not self.trainer_py:
            return
        name = self.model_name_edit.text().strip() or "model"
        yaml_path = getattr(self, "train_yaml", os.path.join(_base_dir(), "datasets", f"{name}.yaml"))
        if not os.path.exists(yaml_path):
            QMessageBox.warning(self, "警告", "先に①データセット作成を実行してください")
            return
        self._train_log("=== 学習開始（数分かかります）===")
        self._run_process(["train_model.py", yaml_path, name])

    def start_export(self):
        """学習済みbest.ptをONNXに変換する（別プロセス）"""
        if not self.trainer_py:
            return
        pt, _ = QFileDialog.getOpenFileName(self, "best.ptを選択",
                                            os.path.join(_base_dir(), "runs"), "PyTorch (*.pt)")
        if not pt:
            return
        name = self.model_name_edit.text().strip() or "model"
        out = os.path.join(_base_dir(), f"{name}.onnx")
        self._train_log("=== ONNX変換開始 ===")
        self._run_process(["export_model.py", pt, out])

    def _run_process(self, args):
        """pythonスクリプトを別プロセスで実行しログを流す"""
        if getattr(self, "_proc", None) is not None and self._proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "情報", "処理が実行中です。完了までお待ちください")
            return
        for b in self.train_buttons:
            b.setEnabled(False)
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(_base_dir())
        self._proc.setProcessChannelMode(QProcess.MergedChannels)
        self._proc.readyReadStandardOutput.connect(
            lambda: self._train_log(bytes(self._proc.readAllStandardOutput()).decode("utf-8", "ignore")))
        self._proc.finished.connect(self._on_proc_finished)
        # スクリプトはbase_dir（pyなら repo、exeなら exe同階層）にある
        script = os.path.join(_base_dir(), args[0])
        self._proc.start(self.trainer_py, [script] + args[1:])

    def _on_proc_finished(self):
        """別プロセス完了時にボタンを戻す"""
        for b in self.train_buttons:
            b.setEnabled(True)
        self._train_log("=== 完了 ===")
        self.set_status("処理が完了しました")

    def _build_inference_tab(self):
        """タブ④：ONNX推論パネルを組み立てる（onnxruntime・ultralytics不使用）"""
        self.model_label = QLabel("モデル未読込")
        self.model_label.setWordWrap(True)

        btn_load = QPushButton("モデル(.onnx)を読み込み")
        btn_load.clicked.connect(self.load_model)

        self.conf_spin = QSpinBox()
        self.conf_spin.setRange(1, 99)
        self.conf_spin.setValue(30)
        self.conf_spin.setSuffix(" %")

        self.infer_cam_index = QSpinBox()
        self.infer_cam_index.setRange(0, 10)

        btn_infer = QPushButton("カメラで推論 開始/停止")
        btn_infer.setFocusPolicy(Qt.NoFocus)
        btn_infer.clicked.connect(self.toggle_inference)

        btn_infer_img = QPushButton("表示中の画像で推論")
        btn_infer_img.setFocusPolicy(Qt.NoFocus)
        btn_infer_img.clicked.connect(self.infer_current_image)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("推論モデル"))
        lay.addWidget(self.model_label)
        lay.addWidget(btn_load)
        lay.addWidget(QLabel("信頼度しきい値"))
        lay.addWidget(self.conf_spin)
        lay.addWidget(QLabel("カメラ番号"))
        lay.addWidget(self.infer_cam_index)
        lay.addWidget(btn_infer)
        lay.addWidget(btn_infer_img)
        lay.addWidget(QLabel("※ best.pt は export_onnx.py で\n   .onnx に変換して読み込みます"))
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def load_model(self):
        """ONNXモデルを選んで検出器を初期化する"""
        path, _ = QFileDialog.getOpenFileName(self, "ONNXモデルを開く", "", "ONNX (*.onnx)")
        if not path:
            return
        try:
            self.detector = OnnxDetector(path, conf=self.conf_spin.value() / 100.0)
            self.model_label.setText(f"読込済: {os.path.basename(path)}")
            self.set_status(f"モデルを読み込みました: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "警告", f"モデル読込失敗: {e}")

    def toggle_inference(self):
        """カメラ推論の開始/停止を切り替える"""
        if self.infer_timer.isActive():
            self._stop_inference()
        else:
            self._start_inference()

    def _start_inference(self):
        """カメラを開いてフレームごとに推論を実行する"""
        if self.detector is None:
            QMessageBox.information(self, "情報", "先にモデル(.onnx)を読み込んでください")
            return
        self._stop_camera()  # 通常のカメラ表示を止める
        idx = self.infer_cam_index.value()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            QMessageBox.warning(self, "警告", f"カメラ {idx} を開けません")
            self.cap = None
            return
        self.detector.conf = self.conf_spin.value() / 100.0
        self.canvas.overlay_text = ""
        self.canvas.reset_view()
        self.canvas.display_mode = "inference"   # 推論モードを明示
        self.canvas.set_live(True)
        self.view_stack.setCurrentIndex(0)
        self.infer_timer.start(50)  # 約20fps（推論は重いので控えめ）
        self.set_status("推論中：カメラ映像をリアルタイム検出 / もう一度押すと停止")

    def _stop_inference(self):
        """推論を停止してカメラを解放する"""
        self.infer_timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.canvas.set_live(False)
        self.canvas.detections = []
        self.canvas.update()
        self.set_status("推論を停止しました")

    def _infer_frame(self):
        """カメラから1フレーム取得して推論・描画する"""
        if self.cap is None or self.detector is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        self.canvas.image = frame
        self.canvas.detections = self.detector.detect(frame)
        self.canvas.update()
        self.set_status(f"推論中：検出 {len(self.canvas.detections)} 個")

    def infer_current_image(self):
        """表示中の静止画に対して推論する"""
        if self.detector is None:
            QMessageBox.information(self, "情報", "先にモデル(.onnx)を読み込んでください")
            return
        if self.canvas.image is None:
            QMessageBox.information(self, "情報", "推論する画像がありません")
            return
        self.detector.conf = self.conf_spin.value() / 100.0
        self.canvas.detections = []                 # 前回の検出枠をクリア
        self.canvas.display_mode = "inference"      # 静止画推論でも検出枠を出す
        self.canvas.detections = self.detector.detect(self.canvas.image)
        self.canvas.update()
        self.set_status(f"推論結果：検出 {len(self.canvas.detections)} 個")

    def keyPressEvent(self, event):
        """ショートカット：Space撮影 / Ctrl+C停止 / Ctrl+S保存 / A,D画像 / W表示 / F1〜クラス"""
        key = event.key()
        ctrl = event.modifiers() == Qt.ControlModifier
        shift = bool(event.modifiers() & Qt.ShiftModifier)  # ガイド微調整用
        # F1〜F12でクラス選択
        if Qt.Key_F1 <= key <= Qt.Key_F12:
            self.select_class(key - Qt.Key_F1)
            return
        if key == Qt.Key_Space:
            self.capture_frame()
        elif key == Qt.Key_C and ctrl:
            self._stop_camera()
        elif key == Qt.Key_S:
            self.save_label()
        elif key == Qt.Key_A:
            self.navigate(-1)
        elif key == Qt.Key_D:
            self.navigate(1)
        elif key == Qt.Key_W:
            self.toggle_view()
        elif key == Qt.Key_K:
            if self.tabs.currentIndex() == 1:   # キーポイント操作はアノテタブのみ
                self.toggle_kpt_mode()
        elif key == Qt.Key_Q:
            if self.tabs.currentIndex() == 1:
                self.canvas.rotate_guide(-0.5 if shift else -1)  # CCW。Shiftで微調整
        elif key == Qt.Key_E:
            if self.tabs.currentIndex() == 1:
                self.canvas.rotate_guide(0.5 if shift else 1)    # CW。Shiftで微調整

    def toggle_kpt_mode(self):
        """キーポイント指定モード（L字頂点を矩形内に1点クリック）のON/OFFを切り替える"""
        new_mode = not self.canvas.kpt_mode
        self.canvas.set_kpt_mode(new_mode)
        if new_mode:
            self.set_status("キーポイントモードON：矩形の内側をクリックで頂点を1点指定 / Kで解除")
        else:
            self.set_status("キーポイントモードOFF：矩形を描画できます / Kで頂点指定")

    def open_file(self):
        """単一画像ファイルを開く"""
        path, _ = QFileDialog.getOpenFileName(self, "画像を開く", "", "Images (*.jpg *.jpeg *.png)")
        if path:
            self._stop_camera()
            self.image_files = [path]
            self.cur_index = 0
            self._load_image(path)

    def open_folder(self):
        """フォルダ内の全画像を読み込んで先頭を表示する"""
        folder = QFileDialog.getExistingDirectory(self, "フォルダを開く")
        if folder:
            self._stop_camera()
            files = []
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                files.extend(glob.glob(os.path.join(folder, ext)))
            if not files:
                QMessageBox.warning(self, "警告", "画像が見つかりません")
                return
            self.image_files = sorted(files)
            self.cur_index = 0
            self._load_image(self.image_files[0])

    def navigate(self, step):
        """フォルダ読込時に前後の画像へ移動する"""
        if not self.image_files:
            return
        self.cur_index = max(0, min(len(self.image_files) - 1, self.cur_index + step))
        self._load_image(self.image_files[self.cur_index])

    def toggle_view(self):
        """単一画面と複数画面（サムネイル一覧）を切り替える"""
        if self.view_stack.currentIndex() == 0:
            if not self.image_files:
                QMessageBox.information(self, "情報", "先に画像を読み込んでください")
                return
            self._populate_gallery()
            self.view_stack.setCurrentIndex(1)
            self.set_status("複数画面：サムネイルをクリックで単一表示 / Wで戻る")
        else:
            self.view_stack.setCurrentIndex(0)
            self.set_status("単一画面：矩形描画 → Ctrl+Sで保存 / A:前 D:次 W:表示切替")

    def _populate_gallery(self):
        """読込中の画像をサムネイルとして一覧に並べる（アノテーション枠付き）"""
        self.gallery.clear()
        for i, path in enumerate(self.image_files):
            pix, annotated, has_kpt = self._make_thumbnail(path)
            if pix is None:
                continue
            mark = " ✓済" if annotated else " －未"
            if has_kpt:
                mark += " K"          # 頂点保存済み
            item = QListWidgetItem(QIcon(pix), os.path.basename(path) + mark)
            item.setData(Qt.UserRole, i)
            self.gallery.addItem(item)

    def _make_thumbnail(self, path):
        """画像に注釈枠・頂点を描いたサムネイルと(注釈有無, 頂点有無)を返す"""
        img = cv2.imread(path)
        if img is None:
            return None, False, False
        h, w = img.shape[:2]
        label_path = os.path.splitext(path)[0] + ".txt"
        annotated = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        has_kpt = False
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    p = line.split()
                    if len(p) not in (5, POSE_NCOLS):   # bbox/Pose 両対応
                        continue
                    cls = int(p[0])
                    cx, cy, bw, bh = map(float, p[1:5])
                    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                    c = class_color(cls)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (c.blue(), c.green(), c.red()), 4)
                    # pose形式なら頂点も十字で描く(v>0 のみ)
                    if len(p) == POSE_NCOLS:
                        v = float(p[7]) if KPT_DIMS == 3 else 2.0
                        if v > 0:
                            has_kpt = True
                            kx, ky = int(float(p[5]) * w), int(float(p[6]) * h)
                            cv2.drawMarker(img, (kx, ky), (255, 0, 255),
                                           cv2.MARKER_CROSS, 16, 3)
        # 頂点が保存済みなら右上に大きく "K" を描く（複数画面で一目で分かる）
        if has_kpt:
            cv2.putText(img, "K", (w - 70, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        2.2, (255, 0, 255), 6, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(260, 195, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix, annotated, has_kpt

    def _on_gallery_click(self, item):
        """サムネイルをクリックしたらその画像を単一画面で開く"""
        idx = item.data(Qt.UserRole)
        self.cur_index = idx
        self._load_image(self.image_files[idx])
        self.view_stack.setCurrentIndex(0)

    def _load_image(self, path):
        """画像を読み込み、既存ラベルがあれば一緒に復元する"""
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "警告", f"読み込み失敗: {path}")
            return
        self.cur_path = path
        self.canvas.set_image(img)
        self.canvas.overlay_text = f"{self.cur_index + 1}/{len(self.image_files)}"  # 右上番号
        self._load_existing_label(path, img.shape[1], img.shape[0])
        self.refresh_list()
        self.set_status(f"{os.path.basename(path)}  ({self.cur_index + 1}/{len(self.image_files)})  A:前 D:次 W:表示切替")

    def _load_existing_label(self, img_path, w, h):
        """同名の.txtがあればYOLO座標を読み込み矩形を復元する。
        bbox(5値) と pose(5+K*D 値) の両方を受理し、pose なら頂点も復元する。
        頂点の v=0(未ラベル) は未指定(None)として扱う。"""
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(label_path):
            self.canvas.keypoints = []
            return
        boxes = []
        keypoints = []
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                # bbox(5) または pose(5+K*D) のみ受理(列数固定で混在を弾く)
                if len(parts) not in (5, POSE_NCOLS):
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((cls, x1, y1, x2, y2))
                if len(parts) == POSE_NCOLS:
                    # 先頭キーポイント(K=1運用)を px,py,v として復元。v=0 は未指定扱い。
                    px = float(parts[5])
                    py = float(parts[6])
                    v = float(parts[7]) if KPT_DIMS == 3 else 2.0
                    keypoints.append((int(px * w), int(py * h)) if v > 0 else None)
                else:
                    keypoints.append(None)
        self.canvas.boxes = boxes
        self.canvas.keypoints = keypoints
        self.canvas.update()

    def toggle_camera(self):
        """カメラの開始と停止を切り替える"""
        if self.cap is None:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        """UVCカメラを開いてライブ映像を開始する"""
        self._stop_inference()  # 推論が動いていれば止める
        idx = self.cam_index.value()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windowsは CAP_DSHOW が安定
        if not self.cap.isOpened():
            QMessageBox.warning(self, "警告", f"カメラ {idx} を開けません")
            self.cap = None
            return
        self.timer.start(30)  # 約33fpsで更新
        self.canvas.overlay_text = ""  # ライブ中は番号を消す
        self.canvas.reset_view()       # ライブはズーム解除
        self.canvas.display_mode = "camera"   # 撮影タブのカメラ起動＝cameraモード
        self.view_stack.setCurrentIndex(0)    # gallery表示中でもライブを前面に
        self.canvas.set_live(True)  # 緑グロー枠ON
        self.set_status(f"カメラ {idx} 起動中：Spaceで撮影 / Ctrl+Cで停止")

    def _stop_camera(self):
        """カメラを停止して解放する"""
        was_running = self.cap is not None
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
        self.canvas.set_live(False)  # 緑グロー枠OFF
        if was_running:
            self.set_status("カメラを停止しました（カメラ開始で再開）")

    def _update_camera(self):
        """カメラから1フレーム取得して表示する（矩形は保持しない）"""
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.canvas.image = frame
            self.canvas.update()

    def capture_frame(self):
        """ライブ映像の現フレームを保存する（カメラは止めず連続撮影できる）"""
        if self.cap is None or self.canvas.image is None:
            return
        frozen = self.canvas.image.copy()
        # カメラもタブも維持。撮影フォルダに連番で保存するだけ
        try:
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            n = len(glob.glob(os.path.join(CAPTURE_DIR, "cap_*.jpg")))
            path = os.path.join(CAPTURE_DIR, f"cap_{n:03d}.jpg")
            cv2.imwrite(path, frozen)
            self.set_status(f"撮影 {n + 1} 枚目を保存: {path}")
        except Exception as e:
            self.set_status(f"保存失敗: {e}")

    def open_capture_folder(self):
        """撮影フォルダ(captures)を開いてアノテーション対象にする"""
        files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "cap_*.jpg")))
        if not files:
            QMessageBox.information(self, "情報", "撮影画像がありません。カメラ撮影タブで撮影してください")
            return
        self._stop_camera()
        self.image_files = files
        self.cur_index = 0
        self._load_image(files[0])

    def refresh_list(self):
        """アノテーション一覧の表示を最新化する（キーポイント有無も付記）"""
        self.box_list.clear()
        kpts = self.canvas.keypoints
        for i, (cls_id, x1, y1, x2, y2) in enumerate(self.canvas.boxes):
            kp = kpts[i] if i < len(kpts) else None
            mark = f"  ◆({kp[0]},{kp[1]})" if kp else "  ◇頂点未指定"
            self.box_list.addItem(f"{class_name(cls_id)}  ({x1},{y1})-({x2},{y2}){mark}")

    def delete_selected(self):
        """一覧で選択中の矩形を削除する"""
        row = self.box_list.currentRow()
        if row >= 0:
            self.canvas.delete_box(row)
            self.refresh_list()

    def save_label(self):
        """現在の矩形をYOLOフォーマット(.txt)で保存する。
        キーポイントが1点でも指定されていれば pose形式(全矩形 5+K*D 列)で保存する。
        頂点未指定の矩形は px=py=0, v=0 で書き、列数を揃える(混在ラベルを作らない)。
        v=0 はultralyticsの損失計算で除外されるため『未ラベル点』として正しく扱われる。"""
        if self.cur_path is None or self.canvas.image is None:
            QMessageBox.warning(self, "警告", "保存対象の画像がありません")
            return
        h, w = self.canvas.image.shape[:2]
        label_path = os.path.splitext(self.cur_path)[0] + ".txt"
        kpts = self.canvas.keypoints
        # 1点でも頂点が指定されていれば pose形式で出す
        pose_mode = any(
            (i < len(kpts) and kpts[i] is not None)
            for i in range(len(self.canvas.boxes))
        )
        with open(label_path, "w") as f:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.canvas.boxes):
                # ピクセル座標 → YOLO正規化座標(cx cy w h)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                if pose_mode:
                    # 全矩形を固定列数(5+K*D)で書く。本プロジェクトは K=1。
                    kp = kpts[i] if i < len(kpts) else None
                    if kp is not None:
                        px, py = kp[0] / w, kp[1] / h
                        line += f" {px:.6f} {py:.6f} 2"   # v=2 可視
                    else:
                        line += " 0.000000 0.000000 0"     # 未指定: v=0(損失除外)
                f.write(line + "\n")
        if pose_mode:
            missing = sum(
                1 for i in range(len(self.canvas.boxes))
                if i >= len(kpts) or kpts[i] is None
            )
            note = f"  ※頂点未指定 {missing}件は v=0 で保存(学習で除外されます)" if missing else ""
            self.set_status(
                f"保存完了(pose 8値): {label_path}（{len(self.canvas.boxes)}件）{note}"
            )
        else:
            self.set_status(f"保存完了(bbox 5値): {label_path}（{len(self.canvas.boxes)}件）")

    def closeEvent(self, event):
        """ウィンドウを閉じる前にカメラ・推論を解放する"""
        self._stop_inference()
        self._stop_camera()
        event.accept()


def main():
    """アプリケーションを起動する"""
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
