from __future__ import annotations

import os, sys, time, math, argparse, warnings, signal
import threading
from collections import deque
from typing import Dict, Deque, List
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype()", category=UserWarning)

import numpy as np
import cv2
import mediapipe as mp

# ====== Shared state for UI preview ======
_last_frame_lock = threading.Lock()
_last_frame_bgr = None
STOP_REQUESTED = False

# ====== Optional deps ======
try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    pd = None  # type: ignore
    HAVE_PANDAS = False
    print("[WARN] No pandas，output CSV not Excel。install: pip install pandas openpyxl xlsxwriter")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib import gridspec, transforms
    HAVE_MPL = True
except Exception:
    plt = None  # type: ignore
    HAVE_MPL = False
    print("[WARN] No matplotlib，skip figures。install: pip install matplotlib")

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets
    HAVE_PG = True
except Exception:
    HAVE_PG = False
    pg = None
    QtWidgets = None

# ====== Keyboard backends ======
if sys.platform == "darwin":
    HAVE_KB2 = False
    kb2 = None
    print("[INFO] macOS")
else:
    try:
        import keyboard as kb2  # type: ignore
        HAVE_KB2 = True
    except Exception:
        HAVE_KB2 = False
        kb2 = None
        print("[WARN] unable to find keyboard library，back to pynput。recommend install: pip install keyboard")

try:
    from pynput.keyboard import Controller, Key  # type: ignore
    kb = Controller()
    HAVE_PYNPUT = True
except Exception:
    kb = None
    HAVE_PYNPUT = False
    print("[WARN] unable to find pynput，。can install: pip install pynput")

# ====== Math helpers ======
def _now() -> float: return time.monotonic()

def angle_abc(a, b, c) -> float:
    a = np.array(a, float); b = np.array(b, float); c = np.array(c, float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 180.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def calculate_velocity(current_pos, previous_pos, dt):
    if previous_pos is None or dt <= 0: return 0.0
    return np.linalg.norm(np.array(current_pos) - np.array(previous_pos)) / dt

def get_landmark_confidence(landmark):
    return landmark.visibility if hasattr(landmark, 'visibility') else 1.0

def smooth_angle(angle_history, new_angle, window_size=3):
    angle_history.append(new_angle)
    if len(angle_history) > window_size: angle_history.pop(0)
    return sum(angle_history) / len(angle_history)

# ====== Keyboard helpers ======
def _press_tap(char: str, label: str) -> None:
    if HAVE_KB2:
        kb2.press(char); time.sleep(0.02); kb2.release(char)
    elif HAVE_PYNPUT and kb:
        keyobj = char if len(char) == 1 else getattr(Key, char)
        kb.press(keyobj); kb.release(keyobj)
    print(f"[KEY] {label}")

def press_space(): _press_tap('space', 'SPACE')
def press_down():  _press_tap('down',  'DOWN')  # keep interface

def key_down(key: str) -> None:
    if HAVE_KB2: kb2.press(key)
    elif HAVE_PYNPUT and kb:
        try: kb.press(key if len(key)==1 else getattr(Key, key))
        except Exception: pass

def key_up(key: str) -> None:
    if HAVE_KB2: kb2.release(key)
    elif HAVE_PYNPUT and kb:
        try: kb.release(key if len(key)==1 else getattr(Key, key))
        except Exception: pass

KEY_SLIDE='s'; KEY_BLOCK='d'; KEY_BLAST='x'

# ================= Motion Summary =================
class MotionLogger:
    ACTIONS = ["Space", "Slide", "Block", "Blast"]
    ACTION_MAP = [
        {"action": "Space", "display_label": "Jump",  "description": "Unilateral Hip and Knee Flexion"},
        {"action": "Slide", "display_label": "Slide", "description": "Bilateral Hip and Knee Flexion"},
        {"action": "Block", "display_label": "Block", "description": "Shoulder Abduction + Elbow Flexion"},
        {"action": "Blast", "display_label": "Blast", "description": "Shoulder Abduction + Elbow Extension"},
    ]
    JOINT_COLS = ["knee_l","knee_r","hip_l","hip_r","elbow_l","elbow_r","shoulder_l","shoulder_r"]
    EXTRA_COLS = ["crouch"]
    ALL_ANGLE_COLS = JOINT_COLS + EXTRA_COLS

    def __init__(self, root_dir: str = "Data Recording"):
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.root_dir = root_dir
        self.out_dir = os.path.join(root_dir, ts)
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"[INFO] 本次数据输出目录: {self.out_dir}")
        self.start_wall = time.time()
        self.rows: List[Dict[str, float]] = []
        self.timeseries: List[Dict[str, float]] = []
        self.counts: Dict[str, int] = {a: 0 for a in self.ACTIONS}
        self.rom_min: Dict[str, float] = {k: float("inf") for k in self.JOINT_COLS}
        self.rom_max: Dict[str, float] = {k: float("-inf") for k in self.JOINT_COLS}

    def log_frame(self, wall_time: float, joints: Dict[str, float]):
        row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time)),
               "t_rel_sec": float(wall_time - self.start_wall)}
        for k in self.ALL_ANGLE_COLS:
            row[k] = float(joints.get(k, np.nan))
        self.timeseries.append(row)
        for k in self.JOINT_COLS:
            v = float(joints.get(k, np.nan))
            if np.isfinite(v):
                if v < self.rom_min[k]: self.rom_min[k] = v
                if v > self.rom_max[k]: self.rom_max[k] = v

    def log(self, action: str, wall_time: float, metrics: Dict[str, float]):
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_time))
        row = {"timestamp": ts_str, "action": action}
        for k in self.ALL_ANGLE_COLS:
            row[k] = float(metrics.get(k, np.nan))
        row["confidence"] = float(metrics.get("confidence", np.nan))
        self.rows.append(row)
        if action in self.counts: self.counts[action] += 1

    def finalize(self):
        total_seconds = round(time.time() - self.start_wall, 2)
        if not (self.rows or self.timeseries):
            print("[INFO] No data for output"); return

        # ---- tables ----
        if HAVE_PANDAS:
            df_raw = pd.DataFrame(self.rows) if self.rows else pd.DataFrame(
                columns=["timestamp","action",*self.ALL_ANGLE_COLS,"confidence"]
            )
            df_ts  = pd.DataFrame(self.timeseries) if self.timeseries else pd.DataFrame(
                columns=["timestamp","t_rel_sec",*self.ALL_ANGLE_COLS]
            )

            # counts + display mapping
            df_counts = pd.DataFrame({
                "action":       [m["action"] for m in self.ACTION_MAP],
                "display_label":[m["display_label"] for m in self.ACTION_MAP],
                "description":  [m["description"] for m in self.ACTION_MAP],
                "count":        [int(self.counts[m["action"]]) for m in self.ACTION_MAP],
            })

            # ROM
            rom_rows=[]
            for j in self.JOINT_COLS:
                jmin = self.rom_min[j] if self.rom_min[j]!=float("inf") else np.nan
                jmax = self.rom_max[j] if self.rom_max[j]!=float("-inf") else np.nan
                rom_rows.append({"joint":j,"min_deg":jmin,"max_deg":jmax,
                                 "range_deg": (jmax-jmin) if (np.isfinite(jmin) and np.isfinite(jmax)) else np.nan})
            df_rom = pd.DataFrame(rom_rows)

            # mean per action-joint
            def _m(frame, col): return float(frame[col].mean()) if (not frame.empty and col in frame) else np.nan
            space=df_raw[df_raw["action"]=="Space"]; slide=df_raw[df_raw["action"]=="Slide"]
            block=df_raw[df_raw["action"]=="Block"]; blast=df_raw[df_raw["action"]=="Blast"]
            aj_rows=[]
            aj_rows += [{"action":"Jump(Space)","joint":"Hip","side":"L","mean_deg":_m(space,"hip_l")},
                        {"action":"Jump(Space)","joint":"Hip","side":"R","mean_deg":_m(space,"hip_r")}]
            aj_rows += [{"action":"Slide","joint":"Knee","side":"L","mean_deg":_m(slide,"knee_l")},
                        {"action":"Slide","joint":"Knee","side":"R","mean_deg":_m(slide,"knee_r")},
                        {"action":"Slide","joint":"Hip","side":"L","mean_deg":_m(slide,"hip_l")},
                        {"action":"Slide","joint":"Hip","side":"R","mean_deg":_m(slide,"hip_r")}]
            aj_rows += [{"action":"Block","joint":"Shoulder","side":"L","mean_deg":_m(block,"shoulder_l")},
                        {"action":"Block","joint":"Shoulder","side":"R","mean_deg":_m(block,"shoulder_r")},
                        {"action":"Block","joint":"Elbow","side":"L","mean_deg":_m(block,"elbow_l")},
                        {"action":"Block","joint":"Elbow","side":"R","mean_deg":_m(block,"elbow_r")}]
            aj_rows += [{"action":"Blast","joint":"Shoulder","side":"L","mean_deg":_m(blast,"shoulder_l")},
                        {"action":"Blast","joint":"Shoulder","side":"R","mean_deg":_m(blast,"shoulder_r")},
                        {"action":"Blast","joint":"Elbow","side":"L","mean_deg":_m(blast,"elbow_l")},
                        {"action":"Blast","joint":"Elbow","side":"R","mean_deg":_m(blast,"elbow_r")}]
            df_ajm=pd.DataFrame(aj_rows)
            df_meta=pd.DataFrame([{"total_duration_sec": total_seconds}])

            xlsx = os.path.join(self.out_dir,"motion_summary.xlsx")
            with pd.ExcelWriter(xlsx, engine=("xlsxwriter" if _has_xlsxwriter() else "openpyxl")) as xw:
                df_meta.to_excel(xw, index=False, sheet_name="meta")
                df_raw.to_excel(xw, index=False, sheet_name="raw")
                df_counts.to_excel(xw, index=False, sheet_name="counts")
                df_rom.to_excel(xw, index=False, sheet_name="rom")
                df_ajm.to_excel(xw, index=False, sheet_name="action_joint_means")
                df_ts.to_excel(xw, index=False, sheet_name="timeseries")
            print(f"[SAVE] {xlsx}")
        else:
            import csv
            def _write_csv(path: str, rows: List[Dict[str, float | str]], header: List[str]) -> None:
                with open(path,"w",newline="",encoding="utf-8") as f:
                    w=csv.DictWriter(f, fieldnames=header); w.writeheader()
                    for r in rows: w.writerow({k:r.get(k,"") for k in header})

            _write_csv(os.path.join(self.out_dir,"meta.csv"), [{"total_duration_sec": total_seconds}], ["total_duration_sec"])
            _write_csv(os.path.join(self.out_dir,"raw.csv"), self.rows, ["timestamp","action",*self.ALL_ANGLE_COLS,"confidence"])
            _write_csv(os.path.join(self.out_dir,"timeseries.csv"), self.timeseries, ["timestamp","t_rel_sec",*self.ALL_ANGLE_COLS])

            counts_rows=[]
            for m in self.ACTION_MAP:
                counts_rows.append({
                    "action": m["action"],
                    "display_label": m["display_label"],
                    "description": m["description"],
                    "count": int(self.counts[m["action"]]),
                })
            _write_csv(os.path.join(self.out_dir,"counts.csv"), counts_rows, ["action","display_label","description","count"])
            print("[SAVE] CSV 导出完成")

        # ---- dashboard (Counts | ROM; Average) ----
        if HAVE_MPL:
            
            COLORS_ACTION = {
                "Space":  "#4C78A8",  # Hip/Jump
                "Slide":  "#F58518",  # Knee/Slide
                "Block":  "#54A24B",  # Shoulder/Block
                "Blast":  "#E45756",  # Elbow/Blast
            }
            COLORS_JOINTTYPE = {
                "hip":      COLORS_ACTION["Space"],
                "knee":     COLORS_ACTION["Slide"],
                "shoulder": COLORS_ACTION["Block"],
                "elbow":    COLORS_ACTION["Blast"],
            }

            # —— 顶parameters——
            LEGEND_Y = 0.895            
            LEFT_TEXT_X = 0.175         
            RIGHT_LEGEND_X = 0.7        

            labels_action = [m["action"] for m in self.ACTION_MAP]
            labels_display = [m["display_label"] for m in self.ACTION_MAP]
            labels_desc    = [m["description"]   for m in self.ACTION_MAP]
            counts_vals    = [int(self.counts[a]) for a in labels_action]
            bar_colors     = [COLORS_ACTION[a] for a in labels_action]

            # ---（Counts ：Jump, Slide, Block, Blast）---
            desired_order = ["Space", "Slide", "Block", "Blast"]  # Space==Jump
            display_map = {a: d for a, d in zip(labels_action, labels_display)}
            desc_map    = {a: d for a, d in zip(labels_action, labels_desc)}
            count_map   = {a: c for a, c in zip(labels_action, counts_vals)}
            color_map   = {a: c for a, c in zip(labels_action, bar_colors)}
            labels_action = desired_order
            labels_display = [display_map[a] for a in desired_order]
            labels_desc    = [desc_map[a]    for a in desired_order]
            counts_vals    = [count_map[a]   for a in desired_order]
            bar_colors     = [color_map[a]   for a in desired_order]

            def jt_type(name:str)->str:
                if name.startswith("hip"): return "hip"
                if name.startswith("knee"): return "knee"
                if name.startswith("shoulder"): return "shoulder"
                return "elbow"

            def pretty_name(j:str)->str:
                base, side = j.split("_")
                return f"{base.capitalize()}_{side.upper()}"

            joints_order = ["hip_l","hip_r","knee_l","knee_r","shoulder_l","shoulder_r","elbow_l","elbow_r"]

            # ========= row for Average） =========
            def _finite(x) -> bool:
                return (x is not None) and np.isfinite(x)

            def _flip180(x):
                return 180.0 - x if _finite(x) else x  

            rows_adj = []
            for r in self.rows:
                rr = dict(r)
                rr["hip_l"]   = _flip180(rr.get("hip_l"))
                rr["hip_r"]   = _flip180(rr.get("hip_r"))
                rr["elbow_l"] = _flip180(rr.get("elbow_l"))
                rr["elbow_r"] = _flip180(rr.get("elbow_r"))
                rows_adj.append(rr)

            # ========= ROM min/max =========
            new_rom_min, new_rom_max = {}, {}
            for j in joints_order:
                omin = self.rom_min[j]
                omax = self.rom_max[j]
                if jt_type(j) in ("hip", "elbow"):
                    if _finite(omin) and _finite(omax):
                        new_rom_min[j] = 180.0 - omax
                        new_rom_max[j] = 180.0 - omin
                    else:
                        new_rom_min[j] = np.nan
                        new_rom_max[j] = np.nan
                else:
                    new_rom_min[j] = omin if _finite(omin) else np.nan
                    new_rom_max[j] = omax if _finite(omax) else np.nan

            # Record joint min/max
            rom_info=[]
            for j in joints_order:
                jmin = new_rom_min[j]
                jmax = new_rom_max[j]
                rng  = (jmax - jmin) if (_finite(jmin) and _finite(jmax)) else 0.0
                rom_info.append((j, jmin, jmax, rng))

            # Average rows_adj
            def _mean_from_rows(rows, col):
                vals=[r.get(col) for r in rows if _finite(r.get(col))]
                return float(np.mean(vals)) if vals else 0.0

            rows_space=[r for r in rows_adj if r.get("action")=="Space"]
            rows_slide=[r for r in rows_adj if r.get("action")=="Slide"]
            rows_block=[r for r in rows_adj if r.get("action")=="Block"]
            rows_blast=[r for r in rows_adj if r.get("action")=="Blast"]
            grouped = [
                ("Space", [("Hip-L",  _mean_from_rows(rows_space,"hip_l")),
                           ("Hip-R",  _mean_from_rows(rows_space,"hip_r"))]),
                ("Slide", [("Knee-L", _mean_from_rows(rows_slide,"knee_l")),
                           ("Knee-R", _mean_from_rows(rows_slide,"knee_r")),
                           ("Hip-L",  _mean_from_rows(rows_slide,"hip_l")),
                           ("Hip-R",  _mean_from_rows(rows_slide,"hip_r"))]),
                ("Block", [("Shoulder-L", _mean_from_rows(rows_block,"shoulder_l")),
                           ("Shoulder-R", _mean_from_rows(rows_block,"shoulder_r")),
                           ("Elbow-L",    _mean_from_rows(rows_block,"elbow_l")),
                           ("Elbow-R",    _mean_from_rows(rows_block,"elbow_r"))]),
                ("Blast", [("Shoulder-L", _mean_from_rows(rows_blast,"shoulder_l")),
                           ("Shoulder-R", _mean_from_rows(rows_blast,"shoulder_r")),
                           ("Elbow-L",    _mean_from_rows(rows_blast,"elbow_l")),
                           ("Elbow-R",    _mean_from_rows(rows_blast,"elbow_r"))]),
            ]

            # Arrangement
            fig = plt.figure(figsize=(13.6, 9.2), dpi=140)
            gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1.22], width_ratios=[1, 1],
                                   hspace=0.40, wspace=0.30)
            fig.subplots_adjust(top=0.80, bottom=0.15)

            title_size = 16

            # --- Counts ---
            ax_c = fig.add_subplot(gs[0,0])
            y_pos = np.arange(len(labels_action))
            ax_c.barh(y_pos, counts_vals, color=bar_colors)
            ax_c.set_xlabel("Count")
            ax_c.set_title("Total Number of Movements", fontsize=title_size, fontweight="bold", pad=14)
            ax_c.set_yticks(y_pos, labels_display)
            for lab in ax_c.get_yticklabels():
                lab.set_fontsize(13) 
            ax_c.invert_yaxis()  # Jump/Slide/Block/Blast
            ax_c.xaxis.set_major_locator(MaxNLocator(integer=True))
            xmax = max(1, max(counts_vals) if counts_vals else 1)
            # Right number show
            offset = 0.015 * xmax
            for i, v in enumerate(counts_vals):
                ax_c.text(v + offset, i, f"{v}", va="center", ha="left")

            # --- ROM ---
            ax_r = fig.add_subplot(gs[0,1], projection="polar")
            ax_r.set_title("Range of Motion (Each Joint)", fontsize=title_size, fontweight="bold", pad=14)
            n = len(joints_order)
            thetas = np.linspace(0, 2*np.pi, n, endpoint=False)
            ax_r.set_theta_zero_location("N"); ax_r.set_theta_direction(-1)
            ax_r.set_xticks([])
            ax_r.set_ylim(0, 180)
            ax_r.set_yticks(np.arange(20, 181, 20))
            ax_r.set_yticklabels([])
            ax_r.grid(True, which="major", axis="y", alpha=0.6)

            outer_r = 186
            LEFT_NAMES  = {"Elbow_R", "Elbow_L", "Shoulder_R"}
            RIGHT_NAMES = {"Hip_R", "Knee_L", "Knee_R"}
            TOP_BOTTOM  = {"Hip_L", "Shoulder_L"}

            for (theta, (jname, jmin, jmax, rng)) in zip(thetas, rom_info):
                color = COLORS_JOINTTYPE[jt_type(jname)]
                is_left = jname.endswith("_l")
                ls = "-" if is_left else "--"
                mk = "o" if is_left else "s"
                
                if _finite(jmin) and _finite(jmax):
                    ax_r.plot([theta, theta], [jmin, jmax], linestyle=ls, marker=mk, color=color, linewidth=2)

                pname = pretty_name(jname)
                if pname in LEFT_NAMES:
                    label_text = f"({jmin:.0f}°–{jmax:.0f}°) {pname}"
                    ha = "right"
                elif pname in RIGHT_NAMES:
                    label_text = f"{pname} ({jmin:.0f}°–{jmax:.0f}°)"
                    ha = "left"
                else:
                    label_text = f"{pname} ({jmin:.0f}°–{jmax:.0f}°)"
                    ha = "center"

                r_text = outer_r + (2 if pname in TOP_BOTTOM else 0)
                ax_r.text(theta, r_text, label_text,
                          ha=ha, va="center", fontsize=10, clip_on=False)

            # --- Average ---
            ax_a = fig.add_subplot(gs[1, :])
            ax_a.set_title("Action-Grouped Average Joint Angles", fontsize=title_size, fontweight="bold", pad=14)
            ax_a.set_ylim(0, 180)  # fixed the range 0-180
            x_positions=[]; heights=[]; tick_labels=[]; colors=[]
            group_spans=[]; x=0.0; gap=1.2
            for action, items in grouped:
                c = COLORS_ACTION[action]; start=x
                for lbl,val in items:
                    x_positions.append(x); heights.append(val); tick_labels.append(lbl); colors.append(c); x += 1.0
                end=x-1.0; group_spans.append((start,end,action)); x += gap
            ax_a.bar(x_positions, heights, color=colors)
            ax_a.set_ylabel("Mean angle (deg)")
            ax_a.set_xticks(x_positions, tick_labels, rotation=28, ha="right")
            trans_g = transforms.blended_transform_factory(ax_a.transData, ax_a.transAxes)
            for (start,end,act) in group_spans:
                center=(start+end)/2.0
                label_txt = "Jump" if act=="Space" else act
                ax_a.text(center, -0.22, label_txt, transform=trans_g, ha="center", va="top",
                          fontsize=12, color="black")
            #top numbers
            for x_i, h in zip(x_positions, heights):
                ax_a.text(x_i, h + 2, f"{h:.1f}", ha="center", va="bottom", fontsize=10, clip_on=False)

            # Top title
            fig.suptitle("Motion Summary Dashboard", fontsize=title_size+6, fontweight="bold",
                         y=0.98, fontfamily="DejaVu Serif")

            # —— Top explaination——
            text_left = (
                "Jump: Unilateral Hip & Knee Flexion\n"
                "Slide: Bilateral Hip & Knee Flexion\n"
                "Block: Shoulder Abduction & Elbow Flexion\n"
                "Blast: Shoulder Abduction & Elbow Extension"
            )
            fig.text(LEFT_TEXT_X, LEGEND_Y, text_left, ha="left", va="center", fontsize=11)

            # Right：ROM 
            from matplotlib.lines import Line2D
            joint_handles = [
                Line2D([0], [0], color=COLORS_JOINTTYPE["hip"], linewidth=4),
                Line2D([0], [0], color=COLORS_JOINTTYPE["knee"], linewidth=4),
                Line2D([0], [0], color=COLORS_JOINTTYPE["shoulder"], linewidth=4),
                Line2D([0], [0], color=COLORS_JOINTTYPE["elbow"], linewidth=4),
            ]
            fig.legend(
                joint_handles, ["Hip", "Knee", "Shoulder", "Elbow"],
                loc="center left", bbox_to_anchor=(RIGHT_LEGEND_X, LEGEND_Y),
                ncol=1, frameon=False, handlelength=2.5, handletextpad=0.8
            )

            # Time of play
            fig.text(0.5, 0.03, f"Total Duration: {total_seconds:.2f} s",
                     ha="center", va="center", fontsize=title_size, fontweight="bold")

            p_all = os.path.join(self.out_dir, "summary_dashboard.png")
            plt.savefig(p_all, bbox_inches="tight")
            plt.close(fig)
            print(f"[SAVE] {p_all}")
        else:
            print("[INFO] matplotlib unavaliable skip photo。")


def _has_xlsxwriter() -> bool:
    try:
        import xlsxwriter  # noqa: F401
        return True
    except Exception:
        return False

# ================= pyqtgraph realtime plotter =================
class AnglePlotterPG:
    """WHY: pyqtgraph real-time low-overhead curves"""
    def __init__(self, window_sec=10.0, refresh_hz=10.0, rescale_hz=1.5):
        if not HAVE_PG: raise RuntimeError("pyqtgraph NA，Pls install：pip install pyqtgraph PyQt5")
        self.window_sec=float(max(2.0, window_sec))
        self.refresh_dt=1.0/max(1.0, float(refresh_hz))
        self.rescale_dt=1.0/max(0.2, float(rescale_hz))
        self.next_update_t=0.0; self.next_rescale_t=0.0
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Angle Trends (pyqtgraph)"); self.win.resize(900,900)
        self.ax_knee=self.win.addPlot(row=0,col=0); self.ax_knee.setLabel('left','Knee (°)'); self.ax_knee.addLegend()
        self.ax_hip=self.win.addPlot(row=1,col=0);  self.ax_hip.setLabel('left','Hip (°)');  self.ax_hip.addLegend()
        self.ax_elbow=self.win.addPlot(row=2,col=0);self.ax_elbow.setLabel('left','Elbow (°)');self.ax_elbow.addLegend()
        self.ax_crouch=self.win.addPlot(row=3,col=0);self.ax_crouch.setLabel('left','Crouch');self.ax_crouch.addLegend(); self.ax_crouch.setLabel('bottom','Time (s)')
        self.curves={"knee_l":self.ax_knee.plot(name="Left"),"knee_r":self.ax_knee.plot(name="Right"),
                     "hip_l":self.ax_hip.plot(name="Left"),  "hip_r":self.ax_hip.plot(name="Right"),
                     "elbow_l":self.ax_elbow.plot(name="Left"),"elbow_r":self.ax_elbow.plot(name="Right"),
                     "crouch":self.ax_crouch.plot(name="Crouch Index")}
        self.sample_hz=20.0; self.maxlen=int(self.window_sec*self.sample_hz)+4
        self.ts:Deque[float]=deque(maxlen=self.maxlen)
        self.buf:Dict[str,Deque[float]]={"knee_l":deque(maxlen=self.maxlen),"knee_r":deque(maxlen=self.maxlen),
                                         "hip_l":deque(maxlen=self.maxlen),"hip_r":deque(maxlen=self.maxlen),
                                         "elbow_l":deque(maxlen=self.maxlen),"elbow_r":deque(maxlen=self.maxlen),
                                         "crouch":deque(maxlen=self.maxlen)}
        for ax in (self.ax_knee,self.ax_hip,self.ax_elbow): ax.setYRange(40,200)
        self.ax_crouch.setYRange(-0.05,0.25); self._last_xrange=(0.0,self.window_sec)
    def _maybe_rescale_y(self, now):
        if now<self.next_rescale_t: return
        self.next_rescale_t=now+self.rescale_dt
        def nzminmax(arrs, default=(0,1)):
            try:
                data=np.concatenate([np.asarray(a,float) for a in arrs if len(a)>0]); data=data[~np.isnan(data)]
                if data.size==0: return default
                lo,hi=float(np.min(data)),float(np.max(data))
                if not np.isfinite(lo) or not np.isfinite(hi) or lo==hi: return default
                pad=(hi-lo)*0.15; return lo-pad, hi+pad
            except Exception: return default
        k_lo,k_hi=nzminmax([self.buf["knee_l"],self.buf["knee_r"]],(60,180))
        h_lo,h_hi=nzminmax([self.buf["hip_l"],self.buf["hip_r"]],(60,180))
        e_lo,e_hi=nzminmax([self.buf["elbow_l"],self.buf["elbow_r"]],(60,180))
        c_lo,c_hi=nzminmax([self.buf["crouch"]],(-0.05,0.25))
        self.ax_knee.setYRange(k_lo,k_hi,padding=0); self.ax_hip.setYRange(h_lo,h_hi,padding=0)
        self.ax_elbow.setYRange(e_lo,e_hi,padding=0); self.ax_crouch.setYRange(c_lo,c_hi,padding=0)
    def update(self, t, values: Dict[str, float]):
        self.ts.append(float(t))
        for k in self.buf.keys(): self.buf[k].append(float(values.get(k, np.nan)))
        if t<self.next_update_t: return
        self.next_update_t=t+self.refresh_dt
        ts=np.fromiter(self.ts,float,count=len(self.ts)); t1=ts[-1] if ts.size else t; t0=t1-self.window_sec
        for k,curve in self.curves.items():
            ys=np.fromiter(self.buf[k],float,count=len(self.buf[k])); curve.setData(ts,ys)
        self._maybe_rescale_y(t)
        if (t0,t1)!=self._last_xrange:
            for ax in (self.ax_knee,self.ax_hip,self.ax_elbow,self.ax_crouch): ax.setXRange(t0,t1,padding=0)
            self._last_xrange=(t0,t1)
        self.app.processEvents()
    def close(self): 
        try: self.win.close()
        except Exception: pass

# ================= Virtual Keyboard =================
class VirtualKeyboard:
    def __init__(self, frame_w, frame_h, scale=1.0):
        self.w, self.h = int(frame_w), int(frame_h)
        self.scale = scale
        gap     = int(14 * scale)
        key_w   = int(90 * scale)
        key_h   = int(54 * scale)
        space_w = int(150 * scale)
        margin  = int(20 * scale)
        self.labels = {"S":"Slide","D":"Block","SPACE":"Jump","X":"Blast"}
        total_w = key_w * 3 + space_w + gap * 3
        base_x = (self.w - total_w) // 2
        base_y = self.h - margin - key_h
        self.rects = {
            "S":     (base_x, base_y, key_w, key_h),
            "D":     (base_x + (key_w + gap), base_y, key_w, key_h),
            "SPACE": (base_x + (key_w + gap) * 2, base_y, space_w, key_h),
            "X":     (base_x + (key_w + gap) * 2 + space_w + gap, base_y, key_w, key_h),
        }
        self.flash_until = {}
        self.hold_keys = set()
        self.col_idle  = (220, 220, 220)
        self.col_flash = (0, 255, 255)
        self.col_hold  = (255, 128, 0)
        self.col_edge  = (40, 40, 40)
        self.col_text  = (0, 0, 0)
        self.font      = cv2.FONT_HERSHEY_SIMPLEX
    def flash(self, key, dur=0.18): self.flash_until[key] = time.time() + float(dur)
    def set_hold(self, key, on=True):
        if on: self.hold_keys.add(key)
        else:  self.hold_keys.discard(key)
    def _draw_key(self, frame, label, rect, state):
        x, y, w, h = rect
        overlay = frame.copy()
        # WHY: different colors indicate states for feedback
        color = self.col_flash if state == 'flash' else (self.col_hold if state == 'hold' else self.col_idle)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.col_edge, 2)
        display_text = self.labels.get(label, label)
        text_scale = 0.8 * self.scale
        (tw, th), _ = cv2.getTextSize(display_text, self.font, text_scale, 2)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2 - 3
        cv2.putText(frame, display_text, (tx, ty), self.font, text_scale, self.col_text, 2, cv2.LINE_AA)
    def draw(self, frame):
        now = time.time()
        for k, rect in self.rects.items():
            state = 'hold' if k in self.hold_keys else ('flash' if self.flash_until.get(k, 0) > now else 'idle')
            self._draw_key(frame, k, rect, state)

# ================= Detections =================
def detect_slide(landmarks, mp_pose, angle_histories=None, *, min_conf: float = 0.7):
    L=mp_pose.PoseLandmark
    lh=landmarks.landmark[L.LEFT_HIP]; rh=landmarks.landmark[L.RIGHT_HIP]
    lk=landmarks.landmark[L.LEFT_KNEE]; rk=landmarks.landmark[L.RIGHT_KNEE]
    la=landmarks.landmark[L.LEFT_ANKLE]; ra=landmarks.landmark[L.RIGHT_ANKLE]
    if (get_landmark_confidence(lh)<min_conf or get_landmark_confidence(rh)<min_conf or
        get_landmark_confidence(lk)<min_conf or get_landmark_confidence(rk)<min_conf): return False
    avg_hip_y=(lh.y+rh.y)/2; avg_knee_y=(lk.y+rk.y)/2; height_ok=(avg_knee_y-avg_hip_y)>0.10
    lk_ang=angle_abc([lh.x,lh.y],[lk.x,lk.y],[la.x,la.y]); rk_ang=angle_abc([rh.x,rh.y],[rk.x,rk.y],[ra.x,ra.y])
    if angle_histories:
        angle_histories.setdefault('slide_left_knee',[]); angle_histories.setdefault('slide_right_knee',[])
        lk_ang=smooth_angle(angle_histories['slide_left_knee'],lk_ang,3); rk_ang=smooth_angle(angle_histories['slide_right_knee'],rk_ang,3)
    return (85<lk_ang<165) and (85<rk_ang<165) and height_ok

def detect_block(landmarks, mp_pose, position_histories=None, *, min_conf: float = 0.6):
    L=mp_pose.PoseLandmark
    sh=landmarks.landmark[L.LEFT_SHOULDER]; el=landmarks.landmark[L.LEFT_ELBOW]; wr=landmarks.landmark[L.LEFT_WRIST]
    for lm in (sh,el,wr):
        if (get_landmark_confidence(lm) or 0.0) < min_conf: return False
    elbow_angle=angle_abc([sh.x,sh.y],[el.x,el.y],[wr.x,wr.y])
    if position_histories is not None:
        hist=position_histories.setdefault('block_hist',{'angle':[],'wr_y':[],'el_y':[],'wr_x':[]})
        hist['angle'].append(float(elbow_angle)); hist['wr_y'].append(float(wr.y)); hist['el_y'].append(float(el.y)); hist['wr_x'].append(float(wr.x))
        N=5
        for k in hist:
            if len(hist[k])>N: hist[k].pop(0)
        elbow_angle=float(np.mean(hist['angle'])); wr_y=float(np.mean(hist['wr_y'])); el_y=float(np.mean(hist['el_y'])); wr_x=float(np.mean(hist['wr_x']))
    else:
        wr_y,el_y,wr_x=wr.y,el.y,wr.x
    angle_ok=(90-25)<=elbow_angle<=(90+25)
    wrist_above=(el_y-wr_y)>0.02
    wrist_near_x=abs(wr_x-sh.x)<0.22
    elbow_near_y=abs(el.y-sh.y)<0.20
    return angle_ok and wrist_above and wrist_near_x and elbow_near_y

def detect_blast(landmarks, mp_pose, velocity_histories=None, *, min_conf: float = 0.6):
    L = mp_pose.PoseLandmark
    rw = landmarks.landmark[L.RIGHT_WRIST]
    re = landmarks.landmark[L.RIGHT_ELBOW]
    rs = landmarks.landmark[L.RIGHT_SHOULDER]

    # only check right-side confidence
    if (get_landmark_confidence(rw) < min_conf or
        get_landmark_confidence(re) < min_conf or
        get_landmark_confidence(rs) < min_conf):
        return False

    # right hand extended forward (keep original logic)
    rf = rw.x < rs.x - 0.15

    # right elbow straightened (keep original logic)
    ra = angle_abc([rs.x, rs.y], [re.x, re.y], [rw.x, rw.y])
    rs_ok = ra > 140

    speed_ok = True
    if velocity_histories:
        now = time.time()
        velocity_histories.setdefault('blast_right_wrist', {'pos': None, 'time': None})

        rspeed = 0
        prev = velocity_histories['blast_right_wrist']
        if prev['pos'] is not None:
            dt = now - prev['time']
            rspeed = calculate_velocity([rw.x, rw.y], prev['pos'], dt)

        velocity_histories['blast_right_wrist']['pos'] = [rw.x, rw.y]
        velocity_histories['blast_right_wrist']['time'] = now

        speed_ok = rspeed > 0.8  # keep original speed threshold

    # only right hand triggers
    return (rf and rs_ok and speed_ok)


# ================= Video I/O =================
def open_capture(source, prefer_backend="auto"):
    if isinstance(source,str) and source.isdigit(): source=int(source)
    plat=sys.platform
    if prefer_backend=="auto":
        if plat.startswith("win"): back=[cv2.CAP_DSHOW, cv2.CAP_MSMF]
        elif plat=="darwin": back=[cv2.CAP_AVFOUNDATION]
        else: back=[cv2.CAP_V4L2]
    else:
        mapping={"dshow":cv2.CAP_DSHOW,"msmf":cv2.CAP_MSMF,"avf":cv2.CAP_AVFOUNDATION,"v4l2":cv2.CAP_V4L2,"auto":0}
        be=mapping.get(prefer_backend.lower()); back=[be] if be is not None else [0]
    if isinstance(source,int):
        for be in back+[cv2.CAP_ANY]:
            try: cap=cv2.VideoCapture(source,be) if be!=cv2.CAP_ANY else cv2.VideoCapture(source)
            except Exception: cap=cv2.VideoCapture(source)
            if cap.isOpened(): print(f"[INFO] opened camera {source} with backend={be}"); return cap
            cap.release()
        raise RuntimeError(f"unable to open camera index {source}")
    cap=cv2.VideoCapture(source)
    if cap.isOpened(): return cap
    raise RuntimeError(f"unabe to open video source: {os.path.abspath(str(source))!r}")

# ================= Main loop =================
def run(source,
        hold_frames=3, cooldown=0.5, prefer_backend="auto",  flip=False,
        knee_flex_thr=120.0, hip_flex_thr=145.0, knee_straight_thr=165.0, height_margin=0.04,
        min_confidence=0.7, adaptive_cooldown=True,
        plot_angles=True, plot_window_sec=10.0, plot_refresh_hz=10.0, show_window=True):
    global STOP_REQUESTED, _last_frame_bgr

    print(f"[INFO] try to open video source: {source}, backend: {prefer_backend}")
    cap=open_capture(source, prefer_backend=prefer_backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
    ok, test_frame = cap.read()
    if not ok: print("[ERROR] unable to read first frame"); return
    h0,w0=test_frame.shape[:2]; vkb=VirtualKeyboard(w0,h0,1.0)

    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    except AttributeError:
        import mediapipe.python.solutions.pose as mp_pose
        import mediapipe.python.solutions.drawing_utils as mp_drawing
    pose=mp_pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    plotter=None
    if plot_angles and HAVE_PG:
        try: plotter=AnglePlotterPG(window_sec=plot_window_sec, refresh_hz=plot_refresh_hz, rescale_hz=1.5)
        except Exception as e: print(f"[WARN] unable to create window：{e}")

    motion_logger=MotionLogger(root_dir="Data Recording")

    streak_leg=streak_slide=streak_block=streak_blast=0
    last_space_t=last_slide_t=last_block_t=last_blast_t=0.0
    angle_histories, position_histories, velocity_histories = {}, {}, {}
    fps_t=_now(); fps_n=0
    error_frames=0; holding_slide=holding_block=holding_blast=False
    off_slide=off_block=off_blast=0

    interrupted = {"flag": False}
    if threading.current_thread() is threading.main_thread():
        def _on_sigint(_s, _f): interrupted["flag"] = True
        signal.signal(signal.SIGINT, _on_sigint)

    try:
        res = None
        while True:
            if STOP_REQUESTED:
                print("[INFO] sopt signal recieved saving image..."); break
            if interrupted["flag"]:
                print("\n[INFO] breaked..."); break

            ok, frame = cap.read()
            if not ok:
                print("[ERR] fail to readout frame"); break

            # ✅ per-frame initialization (must keep)
            l_knee=r_knee=l_hip=r_hip=el_l=el_r=sh_l=sh_r=float("nan")
            avg_conf=0.0

            # ✅ run inference on unflipped frame (keep true left/right)
            h, w = frame.shape[:2]
            t_wall = time.time(); t_mono = _now()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb); fps_n += 1

            # draw landmarks on the unflipped frame
            if res is not None and res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ✅ mirroring for display only: flip at the end
            frame_disp = cv2.flip(frame, 1) if flip else frame

            # overlay UI on the display frame to avoid mirrored text
            vkb.draw(frame_disp)
            with _last_frame_lock:
                _last_frame_bgr = frame_disp.copy()

            # display window
            if show_window:
                cv2.imshow("Pose Game Controller + Motion Summary (pyqtgraph)", frame_disp)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
                    break

            # ===== From here on, your logic remains unchanged =====
            if res.pose_landmarks:
                error_frames=0
                lm=res.pose_landmarks.landmark; L=mp_pose.PoseLandmark
                def P(pt): p=lm[pt.value]; return np.array([p.x*w, p.y*h], float)
                def yN(pt): return lm[pt.value].y
                RS,RH,RK,RA = P(L.RIGHT_SHOULDER), P(L.RIGHT_HIP), P(L.RIGHT_KNEE), P(L.RIGHT_ANKLE)
                LS,LH,LK,LA = P(L.LEFT_SHOULDER),  P(L.LEFT_HIP),  P(L.LEFT_KNEE),  P(L.LEFT_ANKLE)
                r_knee=angle_abc(RH,RK,RA); l_knee=angle_abc(LH,LK,LA)
                r_hip =angle_abc(RS,RH,RK); l_hip =angle_abc(LS,LH,LK)
                l_sh=lm[L.LEFT_SHOULDER.value]; r_sh=lm[L.RIGHT_SHOULDER.value]
                l_el=lm[L.LEFT_ELBOW.value];    r_el=lm[L.RIGHT_ELBOW.value]
                l_wr=lm[L.LEFT_WRIST.value];    r_wr=lm[L.RIGHT_WRIST.value]
                if min(get_landmark_confidence(l_sh), get_landmark_confidence(l_el), get_landmark_confidence(l_wr)) >= 0.5:
                    el_l=angle_abc([l_sh.x,l_sh.y],[l_el.x,l_el.y],[l_wr.x,l_wr.y])
                if min(get_landmark_confidence(r_sh), get_landmark_confidence(r_el), get_landmark_confidence(r_wr)) >= 0.5:
                    el_r=angle_abc([r_sh.x,r_sh.y],[r_el.x,r_el.y],[r_wr.x,r_wr.y])
                l_hip_lm=lm[L.LEFT_HIP.value]; r_hip_lm=lm[L.RIGHT_HIP.value]
                if min(get_landmark_confidence(l_el), get_landmark_confidence(l_sh), get_landmark_confidence(l_hip_lm)) >= 0.5:
                    sh_l=angle_abc([l_el.x,l_el.y],[l_sh.x,l_sh.y],[l_hip_lm.x,l_hip_lm.y])
                if min(get_landmark_confidence(r_el), get_landmark_confidence(r_sh), get_landmark_confidence(r_hip_lm)) >= 0.5:
                    sh_r=angle_abc([r_el.x,r_el.y],[r_sh.x,r_sh.y],[r_hip_lm.x,r_hip_lm.y])
                key_lms=[L.LEFT_HIP,L.RIGHT_HIP,L.LEFT_KNEE,L.RIGHT_KNEE,L.LEFT_ANKLE,L.RIGHT_ANKLE,
                        L.LEFT_SHOULDER,L.RIGHT_SHOULDER,L.LEFT_ELBOW,L.RIGHT_ELBOW,L.LEFT_WRIST,L.RIGHT_WRIST]
                avg_conf = sum(get_landmark_confidence(lm[p.value]) for p in key_lms)/len(key_lms)
                r_knee_y,l_knee_y=yN(L.RIGHT_KNEE),yN(L.LEFT_KNEE); r_ank_y,l_ank_y=yN(L.RIGHT_ANKLE),yN(L.LEFT_ANKLE)
                left_hip_y,right_hip_y=yN(L.LEFT_HIP),yN(L.RIGHT_HIP)


                # per-frame logging
                motion_logger.log_frame(t_wall,{
                    "knee_l":l_knee,"knee_r":r_knee,"hip_l":l_hip,"hip_r":r_hip,
                    "elbow_l":el_l,"elbow_r":el_r,"shoulder_l":sh_l,"shoulder_r":sh_r,
                    "crouch":((l_knee_y+r_knee_y)/2.0 - (left_hip_y+right_hip_y)/2.0),
                })

                # Jump (Space)
                cond_space = ((r_knee < 110.0 and (l_knee_y - r_knee_y) > 0.09 and l_knee > 160.0) or
                              ((l_ank_y - r_ank_y) > 0.11 and l_knee > 160.0))
                streak_leg = streak_leg + 1 if cond_space else 0
                if streak_leg>=hold_frames and (_now()-last_space_t)>cooldown:
                    press_space(); vkb.flash('SPACE',0.20); last_space_t=_now()
                    motion_logger.log("Space",t_wall,{"knee_l":l_knee,"knee_r":r_knee,"hip_l":l_hip,"hip_r":r_hip,
                                                      "elbow_l":el_l,"elbow_r":el_r,"shoulder_l":sh_l,"shoulder_r":sh_r,
                                                      "crouch":((l_knee_y+r_knee_y)/2-(left_hip_y+right_hip_y)/2),"confidence":avg_conf})

                # Slide
                cond_slide=detect_slide(res.pose_landmarks, mp_pose, angle_histories, min_conf=min_confidence)
                if cond_slide: streak_slide+=1; off_slide=0
                else: streak_slide=0; off_slide+=1
                if (not holding_slide) and streak_slide>=hold_frames:
                    key_down(KEY_SLIDE); vkb.set_hold('S',True); holding_slide=True
                    motion_logger.log("Slide",t_wall,{"knee_l":l_knee,"knee_r":r_knee,"hip_l":l_hip,"hip_r":r_hip,
                                                      "elbow_l":el_l,"elbow_r":el_r,"shoulder_l":sh_l,"shoulder_r":sh_r,
                                                      "crouch":((l_knee_y+r_knee_y)/2-(left_hip_y+right_hip_y)/2),"confidence":avg_conf})
                if holding_slide and off_slide>=max(2,hold_frames//2):
                    key_up(KEY_SLIDE); vkb.set_hold('S',False); holding_slide=False

                # Block
                enter_block=detect_block(res.pose_landmarks, mp_pose, position_histories, min_conf=max(0.5,min_confidence-0.1))
                stricter_exit=not enter_block
                hist=position_histories.get('block_hist',None)
                if hist and all(len(hist[k])>=1 for k in ('angle','wr_y','el_y','wr_x')):
                    elbow_angle_sm=float(hist['angle'][-1]); wr_y_sm=float(hist['wr_y'][-1]); el_y_sm=float(hist['el_y'][-1]); wr_x_sm=float(hist['wr_x'][-1])
                    l_sh_x=lm[L.LEFT_SHOULDER.value].x; l_sh_y=lm[L.LEFT_SHOULDER.value].y
                    angle_bad=not ((90-20)<=elbow_angle_sm<=(90+20))
                    wrist_not_above=(el_y_sm-wr_y_sm)<=0.015
                    wrist_x_far=abs(wr_x_sm-l_sh_x)>=0.26
                    elbow_y_far=abs(lm[L.LEFT_ELBOW.value].y-l_sh_y)>=0.22
                    stricter_exit = angle_bad or wrist_not_above or wrist_x_far or elbow_y_far
                if enter_block: streak_block+=1; off_block=0
                else: streak_block=0; off_block+=1
                if (not holding_block) and streak_block>=hold_frames:
                    key_down(KEY_BLOCK); vkb.set_hold('D',True); holding_block=True
                    motion_logger.log("Block",t_wall,{"knee_l":l_knee,"knee_r":r_knee,"hip_l":l_hip,"hip_r":r_hip,
                                                      "elbow_l":el_l,"elbow_r":el_r,"shoulder_l":sh_l,"shoulder_r":sh_r,
                                                      "crouch":((l_knee_y+r_knee_y)/2-(left_hip_y+right_hip_y)/2),"confidence":avg_conf})
                if holding_block:
                    if stricter_exit: off_block+=1
                    else: off_block=0
                    if off_block>=max(3,int(hold_frames)):
                        key_up(KEY_BLOCK); vkb.set_hold('D',False); holding_block=False

                # Blast
                cond_blast=detect_blast(res.pose_landmarks, mp_pose, velocity_histories, min_conf=max(0.5,min_confidence-0.1))
                if cond_blast: streak_blast+=1; off_blast=0
                else: streak_blast=0; off_blast+=1
                if (not holding_blast) and streak_blast>=hold_frames:
                    key_down(KEY_BLAST); vkb.set_hold('X',True); holding_blast=True
                    motion_logger.log("Blast",t_wall,{"knee_l":l_knee,"knee_r":r_knee,"hip_l":l_hip,"hip_r":r_hip,
                                                      "elbow_l":el_l,"elbow_r":el_r,"shoulder_l":sh_l,"shoulder_r":sh_r,
                                                      "crouch":((l_knee_y+r_knee_y)/2-(left_hip_y+right_hip_y)/2),"confidence":avg_conf})
                if holding_blast and off_blast>=max(2,hold_frames//2):
                    key_up(KEY_BLAST); vkb.set_hold('X',False); holding_blast=False

            else:
                error_frames+=1
                if error_frames>=30:
                    if holding_slide: key_up(KEY_SLIDE); vkb.set_hold('S',False); holding_slide=False
                    if holding_block: key_up(KEY_BLOCK); vkb.set_hold('D',False); holding_block=False
                    if holding_blast: key_up(KEY_BLAST); vkb.set_hold('X',False); holding_blast=False

            if show_window:
                cv2.imshow("Pose Game Controller + Motion Summary (pyqtgraph)", frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

    finally:
        try:
            if holding_slide: key_up(KEY_SLIDE); vkb.set_hold('S',False)
            if holding_block: key_up(KEY_BLOCK); vkb.set_hold('D',False)
            if holding_blast: key_up(KEY_BLAST); vkb.set_hold('X',False)
        except Exception: pass
        if plotter: plotter.close()
        cap.release(); cv2.destroyAllWindows()
        try:
            MotionLogger.finalize(motion_logger)
        except Exception as e:
            print(f"[WARN] print Motion Summary fail: {e}")

# ---------- helpers ----------
def start_controller(
    source="0",
    show_window=True, 
    backend="auto",
    flip=True,
    hold_frames=3,
    cooldown=0.5,
    knee_flex_thr=120.0,
    hip_flex_thr=145.0,
    knee_straight_thr=165.0,
    height_margin=0.04,
    min_confidence=0.7,
    adaptive_cooldown=True,
    plot_angles=True,
    plot_window_sec=10.0,
    plot_refresh_hz=10.0,
):
    """Public start entrypoint"""
    global STOP_REQUESTED
    STOP_REQUESTED = False
    src = int(source) if isinstance(source, str) and source.isdigit() else source
    run(source=src,
        hold_frames=hold_frames, cooldown=cooldown, prefer_backend=backend, flip=flip,
        knee_flex_thr=knee_flex_thr, hip_flex_thr=hip_flex_thr, knee_straight_thr=knee_straight_thr,
        height_margin=height_margin, min_confidence=min_confidence, adaptive_cooldown=adaptive_cooldown,
        plot_angles=plot_angles, plot_window_sec=plot_window_sec, plot_refresh_hz=plot_refresh_hz,
        show_window=False)

def stop_controller():
    """Public stop entrypoint"""
    global STOP_REQUESTED
    STOP_REQUESTED = True

def get_last_frame():
    """Return the latest BGR frame; may be None"""
    global _last_frame_bgr
    with _last_frame_lock:
        if _last_frame_bgr is None:
            return None
        return _last_frame_bgr.copy()

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="default camera number（default 0）")
    ap.add_argument("--backend", default="auto", choices=["auto","avf","dshow","msmf","v4l2"], help="backend priority")
    ap.add_argument("--flip", dest="flip", action="store_true", help="mirror（default open）")
    ap.add_argument("--no-flip", dest="flip", action="store_false", help="deactive mirror"); ap.set_defaults(flip=True)
    ap.add_argument("--hold-frames", type=int, default=3)
    ap.add_argument("--cooldown", type=float, default=0.5)
    ap.add_argument("--knee-flex-thr", type=float, default=120.0)
    ap.add_argument("--hip-flex-thr", type=float, default=145.0)
    ap.add_argument("--knee-straight-thr", type=float, default=165.0)
    ap.add_argument("--height-margin", type=float, default=0.04)
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("--adaptive-cooldown", action="store_true", default=True)
    ap.add_argument("--plot-angles", action="store_true", default=True)
    ap.add_argument("--no-plot-angles", dest="plot_angles", action="store_false")
    ap.add_argument("--plot-window-sec", type=float, default=10.0)
    ap.add_argument("--plot-refresh-hz", type=float, default=10.0)
    args = ap.parse_args()
    src = int(args.source) if isinstance(args.source,str) and args.source.isdigit() else args.source
    run(source=src,
        hold_frames=args.hold_frames, cooldown=args.cooldown, prefer_backend=args.backend, flip=args.flip,
        knee_flex_thr=args.knee_flex_thr, hip_flex_thr=args.hip_flex_thr, knee_straight_thr=args.knee_straight_thr,
        height_margin=args.height_margin, min_confidence=args.min_confidence, adaptive_cooldown=args.adaptive_cooldown,
        plot_angles=args.plot_angles, plot_window_sec=args.plot_window_sec, plot_refresh_hz=args.plot_refresh_hz)
