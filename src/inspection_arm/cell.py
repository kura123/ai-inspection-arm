"""検品セル：ベルト、検査位置、良品箱・NG箱、Franka、カメラ2台、録画。

座標（ロボットの台座が原点、単位はm）:
- 部品はベルトの奥（x=0.95）から検査位置（x=0.55, y=0）へ流れてくる
- 良品箱は y=+0.35、NG箱は y=-0.35
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import genesis as gs

STATION = np.array([0.55, 0.0])
BELT_START_X = 0.95
BELT_TOP = 0.01  # ベルト上面の高さ
BINS = {"OK": np.array([0.45, 0.35]), "NG": np.array([0.45, -0.35])}
BIN_INNER = 0.16
BIN_WALL = 0.01
BIN_HEIGHT = 0.06
BIN_COLORS = {"OK": (0.2, 0.4, 0.85), "NG": (0.9, 0.5, 0.1)}
BIN_SLOTS = [(-0.035, -0.035), (0.035, 0.035), (-0.035, 0.035), (0.035, -0.035)]
PARKING = np.array([0.0, 2.0])  # 出番待ちの部品を置く画面外の場所

HAND_DOWN = np.array([0, 1, 0, 0])  # 手先を真下に向ける姿勢
FINGER_OPEN = 0.04
GRIP_FORCE = -20.0
HAND_TO_TIP = 0.11  # hand原点から指先の把持位置までの高さ
# 字幕は日本語。日本語フォント（sudo apt install fonts-noto-cjk）がなければDejaVuで代用する（日本語は豆腐になる）
FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


class Cell:
    def __init__(self, cfg: dict, part_files: list[Path], video_path: Path | None):
        self.cfg = cfg
        self.size = cfg["part"]["size"]
        self.part_z = BELT_TOP + self.size / 2
        gs.init(backend=gs.gpu, logging_level="warning")
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg["sim"]["dt"], substeps=4),
            show_viewer=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self._add_belt()
        for name in BINS:
            self._add_bin(name)
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.parts = []
        for i, f in enumerate(part_files):
            pos = self._parking_pos(i)
            self.parts.append(self.scene.add_entity(gs.morphs.Mesh(file=str(f), pos=pos)))

        insp = cfg["camera"]["inspect"]
        self.inspect_height = BELT_TOP + insp["height"]
        self.inspect_cam = self.scene.add_camera(
            res=tuple(insp["res"]), pos=(*STATION, self.inspect_height),
            lookat=(*STATION, 0), up=(0, 1, 0), fov=insp["fov"],
        )
        ov = cfg["camera"]["overview"]
        self.overview_cam = self.scene.add_camera(
            res=tuple(ov["res"]), pos=(1.65, 1.05, 1.15), lookat=(0.35, 0.0, 0.2), fov=ov["fov"]
        )
        # Step2の方策（ACT）の入力に使うカメラ。手首カメラの姿勢は SortTask が毎周期更新する
        pol = cfg["camera"].get("policy")
        if pol:
            self.front_cam = self.scene.add_camera(
                res=tuple(pol["res"]), pos=tuple(pol["front_pos"]),
                lookat=tuple(pol["front_lookat"]), fov=pol["front_fov"],
            )
            self.wrist_cam = self.scene.add_camera(
                res=tuple(pol["res"]), pos=(0, 0, 1), lookat=(0, 0, 0), fov=pol["wrist_fov"],
                near=pol["wrist_near"],
            )
        self.scene.build()

        self.franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self.franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self.hand = self.franka.get_link("hand")
        self.motors = np.arange(7)
        self.fingers = np.arange(7, 9)
        self.gripping = False
        self.bin_parts = {name: [] for name in BINS}  # 箱に入っている部品の番号

        # 録画（ステップごとではなく一定間隔で全体カメラを描画し、文字を重ねて書き出す）
        self.video_fps = 25
        self.frame_every = max(1, round(1 / (cfg["sim"]["dt"] * self.video_fps)))
        self.n_steps = 0
        self.overlay: dict = {}
        self.writer = imageio.get_writer(video_path, fps=self.video_fps, codec="libx264") if video_path else None
        font = next(f for f in FONTS if Path(f).exists())
        if font != FONTS[0]:
            print(f"warning: 日本語フォントがないため {font} を使う（字幕の日本語は表示されない）")
        self.font = ImageFont.truetype(font, 24)
        self.font_small = ImageFont.truetype(font, 18)

    def _parking_pos(self, i: int):
        return (PARKING[0] + 0.1 * i, PARKING[1], self.part_z)

    # ---- シーンの部品 ----
    def _add_belt(self):
        length = BELT_START_X - STATION[0] + 0.15
        self.scene.add_entity(
            gs.morphs.Box(size=(length, 0.1, BELT_TOP), pos=(STATION[0] + length / 2 - 0.07, 0, BELT_TOP / 2),
                          fixed=True),
            surface=gs.surfaces.Default(color=(0.25, 0.25, 0.28)),
        )

    def _add_bin(self, name: str):
        cx, cy = BINS[name]
        half = BIN_INNER / 2 + BIN_WALL / 2
        long_side = BIN_INNER + 2 * BIN_WALL
        walls = [
            ((cx + half, cy), (BIN_WALL, long_side)),
            ((cx - half, cy), (BIN_WALL, long_side)),
            ((cx, cy + half), (long_side, BIN_WALL)),
            ((cx, cy - half), (long_side, BIN_WALL)),
        ]
        for (x, y), (sx, sy) in walls:
            self.scene.add_entity(
                gs.morphs.Box(size=(sx, sy, BIN_HEIGHT), pos=(x, y, BIN_HEIGHT / 2), fixed=True),
                surface=gs.surfaces.Default(color=BIN_COLORS[name]),
            )

    # ---- 時間を進める・録画 ----
    def step(self):
        if self.gripping:
            self.franka.control_dofs_force(np.array([GRIP_FORCE, GRIP_FORCE]), self.fingers)
        self.scene.step()
        self.n_steps += 1
        if self.writer and self.n_steps % self.frame_every == 0:
            self._write_frame()

    def hold(self, seconds: float):
        """シミュレーションを止めたまま、現在の画面を動画に書き足す（VLMの判定待ちなど）。"""
        if self.writer:
            for _ in range(int(seconds * self.video_fps)):
                self._write_frame()

    def _write_frame(self):
        frame = Image.fromarray(np.asarray(self.overview_cam.render()[0]))
        draw = ImageDraw.Draw(frame)
        lines = self.overlay.get("lines", [])
        if lines:
            draw.rectangle([16, 16, 700, 28 + 32 * len(lines)], fill=(20, 20, 24))
            for k, line in enumerate(lines):
                draw.text((28, 22 + 32 * k), line, font=self.font, fill=(240, 240, 240))
        inset = self.overlay.get("inset")
        if inset is not None:
            w = 240
            x0, y0 = frame.width - w - 20, 20
            frame.paste(inset.resize((w, w)), (x0, y0))
            color = {"OK": (60, 110, 220), "NG": (230, 130, 30)}.get(self.overlay.get("verdict"), (200, 200, 200))
            draw.rectangle([x0 - 4, y0 - 4, x0 + w + 4, y0 + w + 4], outline=color, width=6)
            draw.text((x0, y0 + w + 10), self.overlay.get("inset_label", "検査カメラ"), font=self.font_small, fill=(240, 240, 240))
        self.writer.append_data(np.asarray(frame))

    def close(self):
        if self.writer:
            self.writer.close()

    # ---- 部品の流れ ----
    def feed(self, i: int, steps: int = 100):
        """部品iをベルトの奥から検査位置まで流す。"""
        part = self.parts[i]
        for k in range(steps + 1):
            x = BELT_START_X + (STATION[0] - BELT_START_X) * k / steps
            part.set_pos((x, STATION[1], self.part_z + 0.001), zero_velocity=True)
            self.step()
        for _ in range(20):
            self.step()

    def capture(self) -> Image.Image:
        return Image.fromarray(np.asarray(self.inspect_cam.render()[0]))

    # ---- アームの動作 ----
    def move_to(self, pos, steps: int, settle: int = 0):
        """手先を直交空間で直線補間しながら目標位置へ動かす（経由点ごとにIK）。

        settle: 最後の目標を保ったまま追加で進めるステップ数（PD制御の遅れで届き切らない分を待つ）
        """
        start = self.hand.get_pos().cpu().numpy()
        for k in range(1, steps + settle + 1):
            p = start + (np.asarray(pos) - start) * min(k, steps) / steps
            q = self.franka.inverse_kinematics(link=self.hand, pos=p, quat=HAND_DOWN)
            if self.gripping:
                self.franka.control_dofs_position(q[:7], self.motors)
            else:
                q[-2:] = FINGER_OPEN
                self.franka.control_dofs_position(q)
            self.step()

    def grasp(self, steps: int = 50):
        self.gripping = True
        hold = self.franka.get_dofs_position()[:7]
        for _ in range(steps):
            self.franka.control_dofs_position(hold, self.motors)
            self.step()

    def release(self, steps: int = 40):
        self.gripping = False
        self.franka.control_dofs_position(np.array([FINGER_OPEN, FINGER_OPEN]), self.fingers)
        for _ in range(steps):
            self.step()

    def pick_and_place(self, i: int, bin_name: str) -> dict:
        """検査位置の部品iをつかみ、指定の箱に入れる。把持と仕分けの成否を返す。"""
        part = self.parts[i]
        target = part.get_pos().cpu().numpy()  # Step1ではシミュの真値を使う
        top = self.part_z + self.size / 2
        self.move_to([*target[:2], top + HAND_TO_TIP + 0.15], 100)
        # 指先の把持位置を部品の中心の高さに合わせる
        self.move_to([*target[:2], target[2] + HAND_TO_TIP], 60, settle=30)
        self.grasp()
        self.move_to([*target[:2], top + HAND_TO_TIP + 0.2], 60)
        picked = float(part.get_pos()[2]) > self.part_z + 0.1
        bx, by = BINS[bin_name]
        # 箱の中の4か所に順に置く。4か所が埋まっていたら、先に箱を空にする（満杯の箱を交換する想定）
        if len(self.bin_parts[bin_name]) == len(BIN_SLOTS):
            for k in self.bin_parts[bin_name]:
                self.parts[k].set_pos(self._parking_pos(k), zero_velocity=True)
            self.bin_parts[bin_name] = []
        dx, dy = BIN_SLOTS[len(self.bin_parts[bin_name])]
        self.bin_parts[bin_name].append(i)
        self.move_to([bx + dx, by + dy, BIN_HEIGHT + HAND_TO_TIP + 0.12], 120)
        self.release()
        self.move_to([*STATION, top + HAND_TO_TIP + 0.25], 80)
        for _ in range(20):
            self.step()
        px, py, pz = part.get_pos().cpu().numpy()
        in_bin = (abs(px - bx) < BIN_INNER / 2 and abs(py - by) < BIN_INNER / 2
                  and pz < BIN_HEIGHT + self.size)
        return {"picked": picked, "in_bin": bool(in_bin)}

    def go_home(self, steps: int = 80):
        self.move_to([*STATION, self.part_z + self.size / 2 + HAND_TO_TIP + 0.25], steps)
