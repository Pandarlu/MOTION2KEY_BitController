# file: app.py
import sys
import threading
import cv2
import numpy as np
from pathlib import Path

from PyQt5 import QtWidgets, QtCore, QtGui

import win32gui
import win32con
import win32api

import move_to_key_V10 as ctrl   # your controller module


# Global: last detected external foreground window
_last_foreground_lock = threading.Lock()
_last_foreground = None


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bit Controller")

        # ===== Left side: Camera preview =====
        self.cam_label = QtWidgets.QLabel("No signal yet...")
        self.cam_label.setMinimumSize(600, 400)
        self.cam_label.setAlignment(QtCore.Qt.AlignCenter)

        # ===== Buttons =====
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_snap = QtWidgets.QPushButton("Snap Game")
        self.btn_recover = QtWidgets.QPushButton("Recover")

        # New: flat horizontal button (to show hint image)
        self.btn_hint = QtWidgets.QPushButton("Show Hint")
        self.btn_hint.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.btn_hint.setFixedHeight(56)

        # Assign unique object names for styling
        self.btn_start.setObjectName("btn_start")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_snap.setObjectName("btn_snap")
        self.btn_recover.setObjectName("btn_recover")
        self.btn_hint.setObjectName("btn_hint")

        self.btn_stop.setEnabled(False)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_snap)
        btn_row.addWidget(self.btn_recover)

        # Log output
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(400)

        # Left column layout: camera → four buttons → hint button → log
        left_col = QtWidgets.QVBoxLayout()
        left_col.addWidget(self.cam_label)
        left_col.addLayout(btn_row)
        left_col.addWidget(self.btn_hint)
        left_col.addWidget(self.log)

        # ===== Right side: Snap area (placeholder) =====
        self.game_placeholder = QtWidgets.QLabel("→ Focus the game window, then click Snap")
        self.game_placeholder.setFixedSize(480, 400)
        self.game_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.game_placeholder.setStyleSheet("""
            QLabel {
                background: #000;
                border: 2px dashed #555;
                color: #888;
            }
        """)

        # ===== Layout =====
        root = QtWidgets.QHBoxLayout(self)
        root.addLayout(left_col, stretch=0)
        root.addWidget(self.game_placeholder, stretch=0)

        # ===== Signals =====
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_snap.clicked.connect(self.on_snap)
        self.btn_recover.clicked.connect(self.on_recover)
        self.btn_hint.clicked.connect(self.on_show_hint)

        # Thread handle
        self.ctrl_thread = None

        # Timer: refresh camera frames
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_cam_frame)
        self.timer.start(16)  # ~60fps

        # Timer: periodically record external foreground window
        self.fg_timer = QtCore.QTimer(self)
        self.fg_timer.timeout.connect(self.poll_foreground)
        self.fg_timer.start(150)

        # Apply dark theme
        self.apply_dark_theme()

        # Default window size
        self.resize(1150, 540)

        # Prevent the popup window from being garbage collected
        self._hint_window_ref = None

        # Summary image cache for resize-aware scaling
        self._summary_pixmap: QtGui.QPixmap | None = None

    # ===== Dark Theme =====
    def apply_dark_theme(self):
        dark_qss = """
        QWidget {
            background-color: #121212;
            color: #E0E0E0;
            font-family: "Segoe UI", "Microsoft YaHei";
            font-size: 13px;
        }
        QPushButton {
            border: none;
            border-radius: 8px;
            font-size: 30px;
            font-weight: bold;
            padding: 40px 14px;
            color: #FFFFFF;
        }
        QPushButton#btn_start { background-color: #43A047; }
        QPushButton#btn_start:hover { background-color: #66BB6A; }
        QPushButton#btn_start:pressed { background-color: #2E7D32; }
        QPushButton#btn_start:disabled { background-color: #2a2a2a; color: #777; }

        QPushButton#btn_stop { background-color: #C62828; }
        QPushButton#btn_stop:hover { background-color: #E53935; }
        QPushButton#btn_stop:pressed { background-color: #8E0000; }
        QPushButton#btn_stop:disabled { background-color: #2a2a2a; color: #777; }

        QPushButton#btn_snap { background-color: #7E57C2; }
        QPushButton#btn_snap:hover { background-color: #9575CD; }
        QPushButton#btn_snap:pressed { background-color: #5E35B1; }

        QPushButton#btn_recover { background-color: #283593; }
        QPushButton#btn_recover:hover { background-color: #3949AB; }
        QPushButton#btn_recover:pressed { background-color: #1A237E; }

        QPushButton#btn_hint {
            background-color: #424242;
            font-size: 22px;
            font-weight: 600;
            padding: 12px 16px;
            border-radius: 10px;
        }
        QPushButton#btn_hint:hover { background-color: #616161; }
        QPushButton#btn_hint:pressed { background-color: #212121; }

        QTextEdit {
            background-color: #1E1E1E;
            color: #E0E0E0;
            border: 2px solid #333;
            font-size: 20px;
            font-family: "JetBrains Mono", "Cascadia Code", "Consolas", monospace;
        }
        """
        self.setStyleSheet(dark_qss)

    # ===== Foreground window polling =====
    def poll_foreground(self):
        """Record current foreground window (if not this app) every 150ms."""
        global _last_foreground
        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return
        my_hwnd = int(self.winId())
        if hwnd == my_hwnd:
            return
        with _last_foreground_lock:
            _last_foreground = hwnd

    def get_last_foreground_hwnd(self):
        with _last_foreground_lock:
            return _last_foreground

    # ===== Camera frame update =====
    def update_cam_frame(self):
        frame = ctrl.get_last_frame()
        if frame is None:
            return
        fh, fw, _ = frame.shape
        lw = self.cam_label.width()
        lh = self.cam_label.height()
        scale = min(lw / fw, lh / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                            rgb.strides[0], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.cam_label.setPixmap(pix.scaled(
            self.cam_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

    # ===== Start / Stop =====
    def on_start(self):
        if self.ctrl_thread and self.ctrl_thread.is_alive():
            return
        self.log.append("Controller starting...")
        self.ctrl_thread = threading.Thread(
            target=ctrl.start_controller,
            kwargs={"source": "0"},
            daemon=True
        )
        self.ctrl_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        # 清空右侧可能存在的 summary 显示
        self._summary_pixmap = None
        self.game_placeholder.setText("→ Focus the game window, then click Snap")
        self.game_placeholder.setPixmap(QtGui.QPixmap())

    def on_stop(self):
        """Stop controller, then show latest summary_dashboard.png if available."""
        self.log.append("Controller stopping...")
        try:
            ctrl.stop_controller()
        except Exception as e:
            self.log.append(f"Stop: controller stop error: {e}")

        if self.ctrl_thread and self.ctrl_thread.is_alive():
            self.ctrl_thread.join(timeout=1.5) 
        self.ctrl_thread = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # display
        self.display_latest_summary()

    # ===== Snap =====
    def on_snap(self):
        """
        Attach the last external foreground window to the right placeholder area.
        Goal:
        1. Keep the game window size, only move it.
        2. Match the placeholder size to the game window.
        3. Only resize the main window if it’s too small.
        """
        hwnd = self.get_last_foreground_hwnd()
        if not hwnd:
            self.log.append("Snap: No external window recorded. Please focus the game window first.")
            return

        try:
            g_left, g_top, g_right, g_bottom = win32gui.GetWindowRect(hwnd)
            game_w = g_right - g_left
            game_h = g_bottom - g_top
        except Exception as e:
            self.log.append(f"Snap: Failed to get game window size: {e}")
            return

        self.game_placeholder.setFixedSize(game_w, game_h)

        left_width = 620
        gap = 10
        margin = 20
        needed_w = left_width + gap + game_w + margin

        left_height = 400 + 40 + 90
        needed_h = max(left_height, game_h + 40)

        cur_w = self.width()
        cur_h = self.height()

        new_w, new_h = cur_w, cur_h
        if cur_w < needed_w:
            new_w = needed_w
        if cur_h < needed_h:
            new_h = needed_h

        if new_w != cur_w or new_h != cur_h:
            self.resize(new_w, new_h)
            QtWidgets.QApplication.processEvents()

        placeholder_hwnd = int(self.game_placeholder.winId())
        try:
            ph_left, ph_top, ph_right, ph_bottom = win32gui.GetWindowRect(placeholder_hwnd)
        except Exception as e:
            self.log.append(f"Snap: Could not get placeholder position: {e}")
            return

        target_x, target_y = ph_left, ph_top

        try:
            win32gui.SetWindowPos(
                hwnd, None, target_x, target_y, 0, 0,
                win32con.SWP_NOZORDER | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            )
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        except Exception as e:
            self.log.append(f"Snap: Failed to move game window: {e}")
            return

        title = win32gui.GetWindowText(hwnd)
        self.log.append(
            f"Snap: Moved window '{title}' to ({target_x}, {target_y}), "
            f"placeholder={game_w}x{game_h}, resized=({cur_w}x{cur_h})->({new_w}x{new_h})"
        )

    # ===== Recover =====
    def on_recover(self):
        """If the window flies off-screen, move it back to (100,100) keeping its size."""
        hwnd = self.get_last_foreground_hwnd()
        if not hwnd:
            self.log.append("Recover: No external window recorded.")
            return
        try:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            w, h = r - l, b - t
            win32gui.SetWindowPos(
                hwnd, None, 100, 100, 0, 0,
                win32con.SWP_NOZORDER | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            )
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            title = win32gui.GetWindowText(hwnd)
            self.log.append(f"Recover: Moved '{title}' back to (100,100), keeping size {w}x{h}")
        except Exception as e:
            self.log.append(f"Recover: Failed {e}")

    # ===== Show Hint Image =====
    def on_show_hint(self):
        """Open a resizable window to show hint.jpg (image scales with the window)."""
        try:
            import os
            pix_path = os.path.join(os.path.dirname(__file__), "hint.jpg")
            pix = QtGui.QPixmap(pix_path)
            if pix.isNull():
                self.log.append(f"Hint: Cannot find image: {pix_path}")
                return

            class HintDialog(QtWidgets.QDialog):
                def __init__(self, pm: QtGui.QPixmap, parent=None):
                    super().__init__(parent)
                    self._pixmap = pm

                    self.setWindowTitle("Hint")
                    self.setModal(False)
                    self.resize(900, 650)
                    self.setWindowFlags(
                        QtCore.Qt.Window
                        | QtCore.Qt.WindowCloseButtonHint
                        | QtCore.Qt.WindowMinimizeButtonHint
                        | QtCore.Qt.WindowMaximizeButtonHint
                    )
                    self.setSizeGripEnabled(True)

                    self.scroll = QtWidgets.QScrollArea(self)
                    self.scroll.setWidgetResizable(True)

                    holder = QtWidgets.QWidget()
                    vbox = QtWidgets.QVBoxLayout(holder)
                    vbox.setContentsMargins(0, 0, 0, 0)

                    self.label = QtWidgets.QLabel()
                    self.label.setAlignment(QtCore.Qt.AlignCenter)
                    vbox.addWidget(self.label)

                    self.scroll.setWidget(holder)

                    layout = QtWidgets.QVBoxLayout(self)
                    layout.setContentsMargins(10, 10, 10, 10)
                    layout.addWidget(self.scroll)

                def _update_scaled_pixmap(self):
                    vw = max(1, self.scroll.viewport().width() - 2)
                    vh = max(1, self.scroll.viewport().height() - 2)
                    scaled = self._pixmap.scaled(
                        vw, vh,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation
                    )
                    self.label.setPixmap(scaled)

                def showEvent(self, event):
                    super().showEvent(event)
                    self._update_scaled_pixmap()

                def resizeEvent(self, event):
                    super().resizeEvent(event)
                    self._update_scaled_pixmap()

            dlg = HintDialog(pix, self)
            self._hint_window_ref = dlg
            dlg.show()
            self.log.append("Hint: Hint window opened.")
        except Exception as e:
            self.log.append(f"Hint: Failed to show image: {e}")

    # ===== New: Summary helpers =====
    def find_latest_recording_dir(self, root_dir_name: str = "Data Recording") -> Path | None:
        """Return newest subdir under CWD / root_dir_name, or None if not found."""
        base = Path.cwd() / root_dir_name
        if not base.exists() or not base.is_dir():
            return None
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if not subdirs:
            return None
        subdirs.sort(key=lambda p: p.name, reverse=True)
        return subdirs[0]

    def display_latest_summary(self):
        """Locate and show summary_dashboard.png in a popup dialog (like Hint)."""
        latest = self.find_latest_recording_dir()
        if not latest:
            self.log.append("Summary: 'Data Recording' not found or empty.")
            return
        png_path = latest / "summary_dashboard.png"
        if not png_path.exists():
            self.log.append(f"Summary: Not found: {png_path}")
            return

        pix = QtGui.QPixmap(str(png_path))
        if pix.isNull():
            self.log.append(f"Summary: Failed to load image: {png_path}")
            return

        # —— summary window —— #
        class SummaryDialog(QtWidgets.QDialog):
            def __init__(self, pm: QtGui.QPixmap, parent=None):
                super().__init__(parent)
                self._pixmap = pm
                self.setWindowTitle("Summary")
                self.setModal(False)
                self.resize(1000, 720)
                self.setWindowFlags(
                    QtCore.Qt.Window
                    | QtCore.Qt.WindowCloseButtonHint
                    | QtCore.Qt.WindowMinimizeButtonHint
                    | QtCore.Qt.WindowMaximizeButtonHint
                )
                self.setSizeGripEnabled(True)

                self.scroll = QtWidgets.QScrollArea(self)
                self.scroll.setWidgetResizable(True)

                holder = QtWidgets.QWidget()
                vbox = QtWidgets.QVBoxLayout(holder)
                vbox.setContentsMargins(0, 0, 0, 0)

                self.label = QtWidgets.QLabel()
                self.label.setAlignment(QtCore.Qt.AlignCenter)
                vbox.addWidget(self.label)

                self.scroll.setWidget(holder)

                layout = QtWidgets.QVBoxLayout(self)
                layout.setContentsMargins(10, 10, 10, 10)
                layout.addWidget(self.scroll)

            def _update_scaled_pixmap(self):
                vw = max(1, self.scroll.viewport().width() - 2)
                vh = max(1, self.scroll.viewport().height() - 2)
                scaled = self._pixmap.scaled(
                    vw, vh,
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.label.setPixmap(scaled)

            def showEvent(self, event):
                super().showEvent(event)
                self._update_scaled_pixmap()

            def resizeEvent(self, event):
                super().resizeEvent(event)
                self._update_scaled_pixmap()

        dlg = SummaryDialog(pix, self)
        self._summary_window_ref = dlg   
        dlg.show()

        w, h = pix.width(), pix.height()
        self.log.append(f"Summary: Opened popup {png_path} ({w}x{h})")


    def render_summary_pixmap(self):
        """Scale cached summary to fit placeholder while keeping aspect ratio."""
        if not self._summary_pixmap:
            return
        target_w = max(1, self.game_placeholder.width())
        target_h = max(1, self.game_placeholder.height())
        scaled = self._summary_pixmap.scaled(
            target_w, target_h,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        self.game_placeholder.setPixmap(scaled)

    # Keep summary responsive on window resize
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.render_summary_pixmap()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
