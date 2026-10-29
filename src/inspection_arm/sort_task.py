"""Step2の仕分けタスク：検査位置の部品をつかみ、指定の箱（OK／NG）に入れて手先を戻す。

方策（ACT）とお手本（スクリプト）は同じ形の行動を20Hzで出し、同じ apply_action で実行する。
- 観測: 前方カメラ・手首カメラの画像（256x256）、状態10次元（関節7＋指2＋入れる箱 OK=+1/NG=-1）
- 行動: 8次元（関節の目標角7＋グリッパー：指の目標幅 0.04=開く / 0.0=閉じる）
"""

import numpy as np

from .cell import BIN_HEIGHT, BIN_INNER, BINS, FINGER_OPEN, HAND_DOWN, HAND_TO_TIP, STATION, Cell

STATE_NAMES = [*(f"joint{k}" for k in range(1, 8)), "finger_l", "finger_r", "target_bin"]
ACTION_NAMES = [*(f"joint{k}" for k in range(1, 8)), "gripper"]
GRIP_THRESHOLD = 0.02  # 行動のグリッパー値がこれ未満なら閉じる


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def task_text(bin_name: str) -> str:
    return f"pick the part and place it in the {bin_name} bin"


class SortTask:
    def __init__(self, cell: Cell, cfg: dict):
        self.cell = cell
        self.cfg = cfg["task"]
        self.substeps = round(1 / (cfg["sim"]["dt"] * cfg["sim"]["control_hz"]))
        self.top = cell.part_z + cell.size / 2
        self.home = np.array([*STATION, self.top + HAND_TO_TIP + 0.25])

    # ---- エピソードの開始 ----
    def reset(self, part: int, bin_name: str, rng: np.random.Generator):
        """部品を1つ検査位置に置き（位置と向きはばらつかせる）、アームを開始姿勢にする。"""
        c = self.cell
        for k, p in enumerate(c.parts):
            p.set_pos(c._parking_pos(k), zero_velocity=True)
        xy = STATION + rng.uniform(-1, 1, 2) * self.cfg["pos_noise"]
        yaw = np.deg2rad(rng.uniform(-1, 1) * self.cfg["yaw_noise"])
        c.parts[part].set_pos((*xy, c.part_z + 0.001), zero_velocity=True)
        c.parts[part].set_quat((np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))

        start = self.home + rng.uniform(-1, 1, 3) * self.cfg["home_noise"]
        q = c.franka.inverse_kinematics(link=c.hand, pos=start, quat=HAND_DOWN)
        q[-2:] = FINGER_OPEN
        c.franka.set_dofs_position(q)
        c.franka.control_dofs_position(q)
        c.gripping = False
        self.part = part
        self.bin_name = bin_name
        self.bin_target = BINS[bin_name] + rng.uniform(-1, 1, 2) * self.cfg["bin_noise"]
        for _ in range(20):  # 部品が落ち着くまで待つ
            c.step()

    # ---- 観測と行動 ----
    def observe(self) -> dict:
        c = self.cell
        hp = c.hand.get_pos().cpu().numpy()
        R = quat_to_R(c.hand.get_quat().cpu().numpy())
        # 手首カメラ：手のひらの横（指の開閉方向と直交する向き）から指先の少し先を見る
        c.wrist_cam.set_pose(pos=hp + R @ [0.07, 0, 0.02], lookat=hp + R @ [0, 0, 0.16],
                             up=R @ [1, 0, 0])
        q = c.franka.get_dofs_position().cpu().numpy()
        state = np.concatenate([q, [1.0 if self.bin_name == "OK" else -1.0]]).astype(np.float32)
        return {
            "front": np.asarray(c.front_cam.render()[0]),
            "wrist": np.asarray(c.wrist_cam.render()[0]),
            "state": state,
        }

    def apply_action(self, action):
        c = self.cell
        c.franka.control_dofs_position(np.asarray(action[:7]), c.motors)
        c.gripping = action[7] < GRIP_THRESHOLD
        if not c.gripping:
            c.franka.control_dofs_position(np.array([FINGER_OPEN, FINGER_OPEN]), c.fingers)
        for _ in range(self.substeps):
            c.step()

    # ---- お手本（スクリプト） ----
    def expert_plan(self, estimate: str = "oracle") -> list[tuple[np.ndarray, float]]:
        """手先の目標位置とグリッパー値を周期ごとに並べた計画（部品位置はシミュの真値を使う）。

        estimate は流れ続けるベルト（ConveyorTask）用。部品が止まっているこのタスクでは使わない。
        """
        c = self.cell
        part = c.parts[self.part].get_pos().cpu().numpy()
        start = c.hand.get_pos().cpu().numpy()
        above = np.array([*part[:2], self.top + HAND_TO_TIP + 0.15])
        grasp = np.array([*part[:2], part[2] + HAND_TO_TIP])
        lift = np.array([*part[:2], self.top + HAND_TO_TIP + 0.2])
        drop = np.array([*self.bin_target, BIN_HEIGHT + HAND_TO_TIP + 0.12])
        o, g = FINGER_OPEN, 0.0
        plan: list[tuple[np.ndarray, float]] = [(start, o)] * 4
        for a, b, n, grip in [(start, above, 20, o), (above, grasp, 12, o), (grasp, grasp, 6, o),
                              (grasp, grasp, 10, g), (grasp, lift, 12, g), (lift, drop, 24, g),
                              (drop, drop, 8, o), (drop, self.home, 16, o), (self.home, self.home, 4, o)]:
            plan += [(a + (b - a) * k / n, grip) for k in range(1, n + 1)]
        return plan

    def expert_action(self, target: np.ndarray, grip: float) -> np.ndarray:
        c = self.cell
        q = c.franka.inverse_kinematics(link=c.hand, pos=target, quat=HAND_DOWN).cpu().numpy()
        return np.concatenate([q[:7], [grip]]).astype(np.float32)

    # ---- 成否 ----
    def landed_bin(self) -> str | None:
        """部品が入っている箱の名前（どちらにも入っていなければNone）。"""
        px, py, pz = self.cell.parts[self.part].get_pos().cpu().numpy()
        for name, (bx, by) in BINS.items():
            if (abs(px - bx) < BIN_INNER / 2 and abs(py - by) < BIN_INNER / 2
                    and pz < BIN_HEIGHT + self.cell.size):
                return name
        return None

    def result(self) -> dict:
        landed = self.landed_bin()
        return {"success": landed == self.bin_name, "landed": landed}
