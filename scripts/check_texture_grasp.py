"""初日の確認：部品ごとのテクスチャ、真上カメラの撮影、IKによる立方体の把持。

使い方: uv run python scripts/check_texture_grasp.py
出力: outputs/check_texture_grasp/ に top.png（真上カメラ）と grasp.mp4（把持の様子）
"""

from pathlib import Path

import numpy as np
from PIL import Image

import genesis as gs
from inspection_arm.parts import Defect, export_part

OUT_DIR = Path("outputs/check_texture_grasp")
CUBE = 0.05  # 立方体の一辺 [m]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, substeps=4),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    rng = np.random.default_rng(0)
    defects = {
        "good": Defect("good"),
        "scratch": Defect("scratch", scratch_ratio=0.6),
        "stain": Defect("stain", stain_ratio=0.2),
    }
    cubes = {}
    for (kind, defect), y in zip(defects.items(), [-0.12, 0.0, 0.12]):
        glb = export_part(OUT_DIR / f"{kind}.glb", CUBE, defect, rng)
        cubes[kind] = scene.add_entity(gs.morphs.Mesh(file=str(glb), pos=(0.6, y, CUBE / 2)))
    top_cam = scene.add_camera(res=(512, 512), pos=(0.6, 0.0, 0.6), lookat=(0.6, 0.0, 0.0), fov=40)
    side_cam = scene.add_camera(res=(960, 540), pos=(1.6, 1.0, 0.8), lookat=(0.5, 0.0, 0.2), fov=45)
    scene.build()

    motors = np.arange(7)
    fingers = np.arange(7, 9)
    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]))
    hand = franka.get_link("hand")
    down = np.array([0, 1, 0, 0])  # 手先を真下に向ける姿勢

    # 真上カメラで撮影（テクスチャが部品ごとに違うかの確認）
    rgb = top_cam.render()[0]
    Image.fromarray(np.asarray(rgb)).save(OUT_DIR / "top.png")

    def move_to(pos, steps, finger_width):
        """手先を直交空間で直線補間しながら目標位置へ動かす。"""
        start = hand.get_pos().cpu().numpy()
        for i in range(1, steps + 1):
            p = start + (np.asarray(pos) - start) * i / steps
            q = franka.inverse_kinematics(link=hand, pos=p, quat=down)
            q[-2:] = finger_width
            franka.control_dofs_position(q)
            scene.step()

    target = cubes["scratch"].get_pos().cpu().numpy()
    side_cam.start_recording(save_to_filename=str(OUT_DIR / "grasp.mp4"), fps=50)
    move_to(target + [0, 0, 0.25], 150, 0.04)  # 真上へ
    move_to(target + [0, 0, 0.11], 100, 0.04)  # 降下（hand原点から指先まで約0.1 m）
    for _ in range(50):  # 把持：腕は止めたまま指を力で閉じる
        franka.control_dofs_position(franka.get_dofs_position()[:7], motors)
        franka.control_dofs_force(np.array([-20.0, -20.0]), fingers)
        scene.step()

    start = hand.get_pos().cpu().numpy()
    for i in range(1, 151):  # 持ち上げ（指は力で閉じたまま）
        p = start + np.array([0, 0, 0.2]) * i / 150
        q = franka.inverse_kinematics(link=hand, pos=p, quat=down)
        franka.control_dofs_position(q[:7], motors)
        franka.control_dofs_force(np.array([-20.0, -20.0]), fingers)
        scene.step()
    for _ in range(50):
        scene.step()
    side_cam.stop_recording()

    z = float(cubes["scratch"].get_pos()[2])
    print(f"cube z after lift: {z:.3f} m -> {'GRASP OK' if z > 0.1 else 'GRASP FAILED'}")
    print(f"saved: {OUT_DIR / 'top.png'}, {OUT_DIR / 'grasp.mp4'}")


if __name__ == "__main__":
    main()
