# 作成: 2026-06-28
# 目的: Make Senseのような社内アノテーションツール。
#       UVCライブ映像/静止画から矩形を描き、YOLOフォーマット(.txt)で保存する。

import os
import sys
import glob
import json

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QSize
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QHBoxLayout, QVBoxLayout,
    QSpinBox, QMessageBox, QTabWidget, QStackedWidget, QListView,
    QButtonGroup, QGraphicsDropShadowEffect, QDialog, QPlainTextEdit,
    QDialogButtonBox,
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
        self._scale = 1.0            # 表示倍率
        self._off_x = 0              # 表示時の左オフセット
        self._off_y = 0              # 表示時の上オフセット
        self.live = False            # カメラ起動中フラグ（緑グロー枠の表示用）
        self.overlay_text = ""       # 画像右上に出す番号オーバーレイ（例 "13/20"）

    def set_live(self, flag):
        """カメラ起動中フラグを切り替えて再描画する"""
        self.live = flag
        self.update()

    def set_image(self, bgr):
        """新しい画像をセットして再描画する"""
        self.image = bgr
        self.boxes = []
        self.update()

    def set_class(self, cls_id):
        """描画対象のクラスIDを切り替える"""
        self.current_class = cls_id

    def clear_boxes(self):
        """全ての矩形を削除する"""
        self.boxes = []
        self.update()

    def delete_box(self, index):
        """指定インデックスの矩形を削除する"""
        if 0 <= index < len(self.boxes):
            del self.boxes[index]
            self.update()

    def _calc_transform(self):
        """画像をウィジェットに収める倍率とオフセットを計算する"""
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        self._scale = min(self.width() / w, self.height() / h)
        disp_w, disp_h = int(w * self._scale), int(h * self._scale)
        self._off_x = (self.width() - disp_w) // 2
        self._off_y = (self.height() - disp_h) // 2

    def _widget_to_image(self, pt):
        """ウィジェット座標を元画像ピクセル座標に変換する"""
        x = (pt.x() - self._off_x) / self._scale
        y = (pt.y() - self._off_y) / self._scale
        h, w = self.image.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return int(x), int(y)

    def paintEvent(self, event):
        """画像と全矩形・描画中の矩形を描画する"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        if self.image is None:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "画像またはカメラを読み込んでください")
            return

        self._calc_transform()
        # BGR→RGBに変換してQImageを作る
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        target = QRect(self._off_x, self._off_y, int(w * self._scale), int(h * self._scale))
        painter.drawImage(target, qimg)

        # 保存済み矩形を描画
        for cls_id, x1, y1, x2, y2 in self.boxes:
            self._draw_box(painter, cls_id, x1, y1, x2, y2)

        # ドラッグ中の矩形を描画
        if self.drawing:
            pen = QPen(class_color(self.current_class), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_pt, self.cur_pt))

        # カメラ起動中はウィンドウ枠を蛍光緑でグロー表示する
        if self.live:
            for width, alpha in [(14, 35), (10, 70), (6, 130), (2, 230)]:
                painter.setPen(QPen(QColor(0, 255, 0, alpha), width))
                m = width // 2
                painter.drawRect(self.rect().adjusted(m, m, -m, -m))

        # 単一画面の右上に画像番号をオーバーレイ表示する
        if self.overlay_text and not self.live:
            painter.setFont(QFont("Meiryo", 14, QFont.Bold))
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(self.overlay_text)
            th = fm.height()
            pad = 6
            disp_w = int(self.image.shape[1] * self._scale)
            x = self._off_x + disp_w - tw - pad * 2 - 8
            y = self._off_y + 8
            painter.fillRect(x, y, tw + pad * 2, th + pad, QColor(0, 0, 0, 160))
            painter.setPen(QColor(0, 255, 0))
            painter.drawText(x + pad, y + th - 2, self.overlay_text)

    def _draw_box(self, painter, cls_id, x1, y1, x2, y2):
        """1つの矩形とクラス名ラベルを描画する"""
        color = class_color(cls_id)
        wx1 = int(x1 * self._scale) + self._off_x
        wy1 = int(y1 * self._scale) + self._off_y
        wx2 = int(x2 * self._scale) + self._off_x
        wy2 = int(y2 * self._scale) + self._off_y
        painter.setPen(QPen(color, 2))
        painter.drawRect(QRect(QPoint(wx1, wy1), QPoint(wx2, wy2)))
        painter.setFont(QFont("Meiryo", 9))
        painter.drawText(wx1, wy1 - 4, class_name(cls_id))

    def mousePressEvent(self, event):
        """ドラッグ開始：矩形の始点を記録する"""
        if self.image is None or event.button() != Qt.LeftButton:
            return
        self.drawing = True
        self.start_pt = event.position().toPoint()
        self.cur_pt = self.start_pt

    def mouseMoveEvent(self, event):
        """ドラッグ中：現在点を更新して再描画する"""
        if self.drawing:
            self.cur_pt = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        """ドラッグ終了：矩形を確定して保存する"""
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
        """タブ切替時にガイド表示とクラスボタンの有効/無効を切り替える"""
        # カメラ撮影中はクラスボタンをグレーアウト（注釈はアノテーションタブでのみ）
        if hasattr(self, "class_buttons"):
            for b in self.class_buttons:
                b.setEnabled(index == 1)
        if index == 0:
            self.set_status("カメラ撮影：カメラ開始 → Spaceで連続撮影 / Ctrl+Cで停止")
        else:
            self.set_status("アノテーション：撮影フォルダを開く → 矩形描画 → Sで保存 / A:前 D:次")

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

    def keyPressEvent(self, event):
        """ショートカット：Space撮影 / Ctrl+C停止 / Ctrl+S保存 / A,D画像 / W表示 / F1〜クラス"""
        key = event.key()
        ctrl = event.modifiers() == Qt.ControlModifier
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
            pix, annotated = self._make_thumbnail(path)
            if pix is None:
                continue
            mark = " ✓済" if annotated else " －未"
            item = QListWidgetItem(QIcon(pix), os.path.basename(path) + mark)
            item.setData(Qt.UserRole, i)
            self.gallery.addItem(item)

    def _make_thumbnail(self, path):
        """画像にアノテーション枠を描いたサムネイル(QPixmap)と注釈有無を返す"""
        img = cv2.imread(path)
        if img is None:
            return None, False
        h, w = img.shape[:2]
        label_path = os.path.splitext(path)[0] + ".txt"
        annotated = os.path.exists(label_path) and os.path.getsize(label_path) > 0
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    p = line.split()
                    if len(p) != 5:
                        continue
                    cls = int(p[0])
                    cx, cy, bw, bh = map(float, p[1:])
                    x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                    x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                    c = class_color(cls)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (c.blue(), c.green(), c.red()), 4)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(260, 195, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix, annotated

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
        """同名の.txtがあればYOLO座標を読み込み矩形を復元する"""
        label_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(label_path):
            return
        boxes = []
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((cls, x1, y1, x2, y2))
        self.canvas.boxes = boxes
        self.canvas.update()

    def toggle_camera(self):
        """カメラの開始と停止を切り替える"""
        if self.cap is None:
            self._start_camera()
        else:
            self._stop_camera()

    def _start_camera(self):
        """UVCカメラを開いてライブ映像を開始する"""
        idx = self.cam_index.value()
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windowsは CAP_DSHOW が安定
        if not self.cap.isOpened():
            QMessageBox.warning(self, "警告", f"カメラ {idx} を開けません")
            self.cap = None
            return
        self.timer.start(30)  # 約33fpsで更新
        self.canvas.overlay_text = ""  # ライブ中は番号を消す
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
        """アノテーション一覧の表示を最新化する"""
        self.box_list.clear()
        for cls_id, x1, y1, x2, y2 in self.canvas.boxes:
            self.box_list.addItem(f"{class_name(cls_id)}  ({x1},{y1})-({x2},{y2})")

    def delete_selected(self):
        """一覧で選択中の矩形を削除する"""
        row = self.box_list.currentRow()
        if row >= 0:
            self.canvas.delete_box(row)
            self.refresh_list()

    def save_label(self):
        """現在の矩形をYOLOフォーマット(.txt)で保存する"""
        if self.cur_path is None or self.canvas.image is None:
            QMessageBox.warning(self, "警告", "保存対象の画像がありません")
            return
        h, w = self.canvas.image.shape[:2]
        label_path = os.path.splitext(self.cur_path)[0] + ".txt"
        with open(label_path, "w") as f:
            for cls_id, x1, y1, x2, y2 in self.canvas.boxes:
                # ピクセル座標 → YOLO正規化座標(cx cy w h)
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = abs(x2 - x1) / w
                bh = abs(y2 - y1) / h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        self.set_status(f"保存完了: {label_path}（{len(self.canvas.boxes)}件）")

    def closeEvent(self, event):
        """ウィンドウを閉じる前にカメラを解放する"""
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
