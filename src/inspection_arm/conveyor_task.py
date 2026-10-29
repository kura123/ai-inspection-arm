"""Step2（発展）：流れ続けるベルトの上の部品を、止めずにつかんで箱に入れる。

- 部品はベルトの上流から一定速度で流れてくる。速度はエピソードごとに変わり、方策には教えない
- 指が部品をはさむまでベルトは止まらない
- 観測の状態は17次元（関節7＋指2＋入れる箱＋関節速度7）。1枚の画像では速さが分からないため関節速度を加える
- お手本と比較用のスクリプトは、部品の位置の求め方だけが違う
  - oracle（お手本）：毎周期、シミュの真の位置と速度を使う
  - nominal（比較用スクリプト）：最初に見た位置と、決められた速度から推定する（速度センサーなし）
"""

import numpy as np

from .cell import BIN_HEIGHT, FINGER_OPEN, HAND_TO_TIP, STATION
from .sort_task import STATE_NAMES as BASE_STATE_NAMES
from .sort_task import GRIP_THRESHOLD, SortTask

STATE_NAMES = [*BASE_STATE_NAMES, *(f"joint{k}_vel" for k in range(1, 8))]
HOLD_WIDTH = 0.0265  # 指がこれより閉じていて、手先が部品の真上にあれば「はさんだ」とみなす
BELT_END_X = STATION[0] - 0.07  # ベルトの下流端


class ConveyorTask(SortTask):
    def __init__(self, cell, cfg: dict):
        super().__init__(cell, cfg)
        self.conv = cfg["task"]["conveyor"]
        self.dt = cfg["sim"]["dt"]
        self.control_dt = self.dt * self.substeps

    def reset(self, part: int, bin_name: str, rng: np.random.Generator, speed: float | None = None):
        super().reset(part, bin_name, rng)
        c = self.cell
        y = rng.uniform(-1, 1) * self.conv["y_noise"]
        c.parts[part].set_pos((self.conv["start_x"], y, c.part_z + 0.001), zero_velocity=True)
        self.speed = rng.uniform(*self.conv["speed"]) if speed is None else speed
        self.on_belt = True
        self.start_xy = np.array([self.conv["start_x"], y])  # 比較用スクリプトが知っている位置

    def observe(self) -> dict:
        obs = super().observe()
        dq = self.cell.franka.get_dofs_velocity().cpu().numpy()[:7]
        obs["state"] = np.concatenate([obs["state"], dq]).astype(np.float32)
        return obs

    def apply_action(self, action):
        c = self.cell
        c.franka.control_dofs_position(np.asarray(action[:7]), c.motors)
        c.gripping = action[7] < GRIP_THRESHOLD
        if not c.gripping:
            c.franka.control_dofs_position(np.array([FINGER_OPEN, FINGER_OPEN]), c.fingers)
        for _ in range(self.substeps):
            self._convey()
            c.step()

    def _convey(self):
        """ベルトの上にある間、部品を下流（-x方向）へ動かす。"""
        if not self.on_belt:
            return
        c = self.cell
        p = c.parts[self.part].get_pos().cpu().numpy()
        fingers = c.franka.get_dofs_position().cpu().numpy()[7:]
        hand = c.hand.get_pos().cpu().numpy()
        held = c.gripping and fingers.max() < HOLD_WIDTH and np.linalg.norm(hand[:2] - p[:2]) < 0.02
        if held or p[2] > c.part_z + 0.01 or p[0] < BELT_END_X:
            self.on_belt = False
            return
        c.parts[self.part].set_pos((p[0] - self.speed * self.dt, p[1], p[2]), zero_velocity=True)

    # ---- お手本（oracle）と比較用スクリプト（nominal） ----
    def part_estimate(self, tick: int, estimate: str) -> np.ndarray:
        """手先を合わせる部品のxy。追従の遅れを補うため、少し先の位置を返す。"""
        lead = self.conv["lead"]
        if estimate == "oracle":
            p = self.cell.parts[self.part].get_pos().cpu().numpy()[:2]
            v = self.speed if self.on_belt else 0.0
        else:
            v = self.conv["nominal_speed"]
            p = self.start_xy - [v * tick * self.control_dt, 0.0]
        return p - [v * lead, 0.0]

    def expert_plan(self, estimate: str = "oracle"):
        """周期ごとに（手先の目標位置, グリッパー値）を返すジェネレータ。

        部品が動くので、計画を先に作らず、周期ごとにその時点の部品の位置から目標を決める。
        """
        c = self.cell
        start = c.hand.get_pos().cpu().numpy()
        hover = self.top + HAND_TO_TIP + 0.15
        grasp_z = c.part_z + HAND_TO_TIP
        drop = np.array([*self.bin_target, BIN_HEIGHT + HAND_TO_TIP + 0.12])
        o, g = FINGER_OPEN, 0.0
        t = 0

        def at(z):
            return np.array([*self.part_estimate(t, estimate), z])

        for _ in range(4):
            yield start, o
            t += 1
        for k in range(1, 21):  # 流れている部品の真上へ
            yield start + (at(hover) - start) * k / 20, o
            t += 1
        for k in range(1, 13):  # 部品に合わせて動きながら下りる
            yield at(hover + (grasp_z - hover) * k / 12), o
            t += 1
        for _ in range(6):
            yield at(grasp_z), o
            t += 1
        for _ in range(10):  # 部品に合わせて動きながら閉じる
            target = at(grasp_z)
            yield target, g
            t += 1
        lift = np.array([*target[:2], self.top + HAND_TO_TIP + 0.2])
        for a, b, n, grip in [(target, lift, 12, g), (lift, drop, 24, g), (drop, drop, 8, o),
                              (drop, self.home, 16, o), (self.home, self.home, 4, o)]:
            for k in range(1, n + 1):
                yield a + (b - a) * k / n, grip


def make_task(cell, cfg: dict) -> SortTask:
    """設定に conveyor があれば流れ続けるベルト、なければ止まった部品のタスクを作る。"""
    return ConveyorTask(cell, cfg) if "conveyor" in cfg["task"] else SortTask(cell, cfg)


def state_names(cfg: dict) -> list[str]:
    return STATE_NAMES if "conveyor" in cfg["task"] else BASE_STATE_NAMES
