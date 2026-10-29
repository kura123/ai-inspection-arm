"""環境の動作確認：GPU、Genesis（Frankaの表示とmp4録画）、Claude APIキーの有無。

使い方: uv run python scripts/check_env.py
"""

import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

OUT_DIR = Path("outputs/check_env")


def check_gpu() -> None:
    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def check_genesis() -> None:
    import genesis as gs

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    cam = scene.add_camera(res=(640, 480), pos=(2.0, 1.5, 1.2), lookat=(0, 0, 0.4), fov=40)
    scene.build()

    # 関節を少しずつ動かしながら録画する（録画中はscene.step()ごとにカメラが自動で描画する）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = OUT_DIR / "franka.mp4"
    cam.start_recording(save_to_filename=str(video), fps=50)
    n_dofs = franka.n_dofs
    for i in range(200):
        target = np.zeros(n_dofs)
        target[0] = 0.8 * np.sin(i / 40)
        target[3] = -1.5
        franka.control_dofs_position(target)
        scene.step()
    cam.stop_recording()
    print(f"Genesis {gs.__version__}: recorded {video}")


def check_api_key() -> None:
    load_dotenv()
    print(f"ANTHROPIC_API_KEY set: {bool(os.environ.get('ANTHROPIC_API_KEY'))}")


if __name__ == "__main__":
    check_gpu()
    check_api_key()
    check_genesis()
