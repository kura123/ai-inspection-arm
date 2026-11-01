"""学習済み方策（ACT、SmolVLA）の読み込みと、シミュの観測をLeRobotの入力形式にする処理。"""

import json
from pathlib import Path

import numpy as np
import torch

IMAGE_KEYS = {"front": "observation.images.front", "wrist": "observation.images.wrist",
              "inspect": "observation.images.inspect"}


def load_policy(path: str, n_action_steps: int | None = None):
    """config.json の type（act / smolvla など）に合う方策と、前処理・後処理を読み込む。"""
    from lerobot.policies import get_policy_class, make_pre_post_processors

    kind = json.loads((Path(path) / "config.json").read_text())["type"]
    policy = get_policy_class(kind).from_pretrained(path).to("cuda").eval()
    if n_action_steps:
        policy.config.n_action_steps = n_action_steps  # 予測し直す間隔は学習し直さずに変えられる
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, pre, post


def to_batch(obs: dict, task: str) -> dict:
    """観測をLeRobotの形式にする（画像は 1xCxHxW、0〜1のfloat）。"""
    def image(a):
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0).float() / 255
    batch = {key: image(obs[name]) for name, key in IMAGE_KEYS.items() if name in obs}
    batch["observation.state"] = torch.from_numpy(obs["state"]).unsqueeze(0)
    batch["task"] = [task]
    return batch


def act_step(policy, pre, post, obs: dict, task: str) -> np.ndarray:
    with torch.inference_mode():
        action = post(policy.select_action(pre(to_batch(obs, task))))
    return action.squeeze(0).cpu().numpy()
