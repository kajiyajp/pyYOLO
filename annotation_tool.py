# 作成: 2026-06-28
# 目的: Make Senseのような社内アノテーションツール。
#       UVCライブ映像/静止画から矩形を描き、YOLOフォーマット(.txt)で保存する。

import os
import sys
import glob

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRect, QPoint
from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QListWidget, QFileDialog, QHBoxLayout, QVBoxLayout, QSpinBox, QMessageBox,
    QTabWidget,
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

# 対象クラス定義（class_id順）
CLASSES = ["l_mark", "cross", "circle_cross", "l_mark_black"]
# クラスごとの描画色（BGRではなくRGB）
CLASS_COLORS = [
    QColor(0, 255, 0),     # l_mark = 緑
    QColor(0, 180, 255),   # cross = 水色
    QColor(255, 180, 0),   # circle_cross = オレンジ
    QColor(255, 0, 180),   # l_mark_black = ピンク
]


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
            pen = QPen(CLASS_COLORS[self.current_class], 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(QRect(self.start_pt, self.cur_pt))

    def _draw_box(self, painter, cls_id, x1, y1, x2, y2):
        """1つの矩形とクラス名ラベルを描画する"""
        color = CLASS_COLORS[cls_id]
        wx1 = int(x1 * self._scale) + self._off_x
        wy1 = int(y1 * self._scale) + self._off_y
        wx2 = int(x2 * self._scale) + self._off_x
        wy2 = int(y2 * self._scale) + self._off_y
        painter.setPen(QPen(color, 2))
        painter.drawRect(QRect(QPoint(wx1, wy1), QPoint(wx2, wy2)))
        painter.setFont(QFont("Meiryo", 9))
        painter.drawText(wx1, wy1 - 4, CLASSES[cls_id])

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
        self.status = QLabel("準備完了")
        self.status.setWordWrap(True)

        # タブで「カメラ撮影」と「アノテーション」を分ける
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_camera_tab(), "カメラ撮影")
        self.tabs.addTab(self._build_annotation_tab(), "アノテーション")

        left = QVBoxLayout()
        left.addWidget(self.tabs)
        left.addWidget(self.status)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(280)

        root = QHBoxLayout()
        root.addWidget(left_widget)
        root.addWidget(self.canvas, stretch=1)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

    def _build_camera_tab(self):
        """タブ①：カメラ撮影パネルを組み立てる"""
        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 10)

        btn_cam = QPushButton("カメラ開始/停止")
        btn_cam.clicked.connect(self.toggle_camera)
        btn_capture = QPushButton("キャプチャ（Space）")
        btn_capture.clicked.connect(self.capture_frame)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("カメラ番号"))
        lay.addWidget(self.cam_index)
        lay.addWidget(btn_cam)
        lay.addWidget(btn_capture)
        lay.addWidget(QLabel("※ キャプチャすると\n   アノテーションタブに切替わります"))
        lay.addStretch(1)
        w = QWidget()
        w.setLayout(lay)
        return w

    def _build_annotation_tab(self):
        """タブ②：アノテーションパネルを組み立てる"""
        self.class_combo = QComboBox()
        self.class_combo.addItems(CLASSES)
        self.class_combo.currentIndexChanged.connect(self.canvas.set_class)

        btn_open_file = QPushButton("画像を開く")
        btn_open_file.clicked.connect(self.open_file)
        btn_open_folder = QPushButton("フォルダを開く")
        btn_open_folder.clicked.connect(self.open_folder)
        btn_prev = QPushButton("← 前の画像")
        btn_prev.clicked.connect(lambda: self.navigate(-1))
        btn_next = QPushButton("次の画像 →")
        btn_next.clicked.connect(lambda: self.navigate(1))
        btn_delete = QPushButton("選択した矩形を削除")
        btn_delete.clicked.connect(self.delete_selected)
        btn_clear = QPushButton("全矩形クリア")
        btn_clear.clicked.connect(self.canvas.clear_boxes)
        btn_save = QPushButton("YOLO形式で保存（Ctrl+S）")
        btn_save.clicked.connect(self.save_label)

        self.box_list = QListWidget()

        lay = QVBoxLayout()
        lay.addWidget(QLabel("クラス選択"))
        lay.addWidget(self.class_combo)
        lay.addWidget(btn_open_file)
        lay.addWidget(btn_open_folder)
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
        """ショートカット：Spaceでキャプチャ、Ctrl+Sで保存"""
        if event.key() == Qt.Key_Space:
            self.capture_frame()
        elif event.key() == Qt.Key_S and event.modifiers() == Qt.ControlModifier:
            self.save_label()

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

    def _load_image(self, path):
        """画像を読み込み、既存ラベルがあれば一緒に復元する"""
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "警告", f"読み込み失敗: {path}")
            return
        self.cur_path = path
        self.canvas.set_image(img)
        self._load_existing_label(path, img.shape[1], img.shape[0])
        self.refresh_list()
        self.status.setText(f"{os.path.basename(path)}  ({self.cur_index + 1}/{len(self.image_files)})")

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
        self.status.setText(f"カメラ {idx} 起動中")

    def _stop_camera(self):
        """カメラを停止して解放する"""
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None

    def _update_camera(self):
        """カメラから1フレーム取得して表示する（矩形は保持しない）"""
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret:
            self.canvas.image = frame
            self.canvas.update()

    def capture_frame(self):
        """ライブ映像の現フレームを静止画として確定する"""
        if self.cap is None or self.canvas.image is None:
            return
        frozen = self.canvas.image.copy()
        self._stop_camera()
        self.canvas.set_image(frozen)
        self.refresh_list()
        self.tabs.setCurrentIndex(1)  # 保存可否に関わらず先にアノテーションタブへ切替える

        # 連番ファイル名で一時保存する（保存失敗してもUIは止めない）
        try:
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            n = len(glob.glob(os.path.join(CAPTURE_DIR, "cap_*.jpg")))
            self.cur_path = os.path.join(CAPTURE_DIR, f"cap_{n:03d}.jpg")
            cv2.imwrite(self.cur_path, frozen)
            self.image_files = [self.cur_path]
            self.cur_index = 0
            self.status.setText(f"キャプチャ保存: {self.cur_path}")
        except Exception as e:
            self.cur_path = None
            self.status.setText(f"キャプチャ表示OK・保存失敗: {e}")

    def refresh_list(self):
        """アノテーション一覧の表示を最新化する"""
        self.box_list.clear()
        for cls_id, x1, y1, x2, y2 in self.canvas.boxes:
            self.box_list.addItem(f"{CLASSES[cls_id]}  ({x1},{y1})-({x2},{y2})")

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
        self.status.setText(f"保存完了: {label_path}（{len(self.canvas.boxes)}件）")

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
