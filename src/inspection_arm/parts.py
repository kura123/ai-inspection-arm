"""部品（立方体）と不良テクスチャの生成。

Genesisの組み込みBoxにはUVがなくテクスチャを貼れないため、UV付きの立方体メッシュを作ってglbで書き出す。
上面（+z）にテクスチャ全体を割り当て、他の面にはテクスチャの縁（地の色だけの領域）を割り当てる。
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

TEX_SIZE = 256
BORDER = 8  # 縁は地の色のまま残し、側面に使う
BASE_COLOR = (205, 205, 200)


@dataclass
class Defect:
    """不良のパラメータ。これがそのまま正解ラベルになる。"""

    kind: str  # "good" | "scratch" | "stain"
    scratch_ratio: float = 0.0  # 傷の長さ / 一辺
    stain_ratio: float = 0.0  # 変色した面積 / 上面の面積
    variant: str = "good"  # good | scratch_small | scratch_large | stain_small | stain_large

    def to_dict(self) -> dict:
        return asdict(self)

    def is_ng(self, thresholds: dict[str, float]) -> bool:
        """基準のしきい値に照らした正解。しきい値0は「少しでもあればNG」。"""
        ratio = {"scratch": self.scratch_ratio, "stain": self.stain_ratio}.get(self.kind, 0.0)
        if self.kind not in thresholds or ratio <= 0:
            return False
        return ratio >= thresholds[self.kind]


# 基準の境界（傷1/3、色ムラ1割）からはっきり離した範囲
VARIANTS = {
    "good": ("good", None),
    "scratch_small": ("scratch", (0.08, 0.15)),
    "scratch_large": ("scratch", (0.6, 0.8)),
    "stain_small": ("stain", (0.015, 0.03)),
    "stain_large": ("stain", (0.25, 0.35)),
}


def sample_defect(variant: str, rng: np.random.Generator) -> Defect:
    kind, ratio_range = VARIANTS[variant]
    ratio = float(rng.uniform(*ratio_range)) if ratio_range else 0.0
    return Defect(
        kind=kind,
        scratch_ratio=ratio if kind == "scratch" else 0.0,
        stain_ratio=ratio if kind == "stain" else 0.0,
        variant=variant,
    )


def make_texture(defect: Defect, rng: np.random.Generator) -> Image.Image:
    img = Image.new("RGB", (TEX_SIZE, TEX_SIZE), BASE_COLOR)
    draw = ImageDraw.Draw(img)
    inner = TEX_SIZE - 2 * BORDER
    if defect.kind == "scratch":
        length = defect.scratch_ratio * inner
        angle = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(BORDER + length / 2, TEX_SIZE - BORDER - length / 2, size=2)
        dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(15, 15, 15), width=6)
    elif defect.kind == "stain":
        # 面積比から円の半径を決める
        radius = np.sqrt(defect.stain_ratio * inner * inner / np.pi)
        cx, cy = rng.uniform(BORDER + radius, TEX_SIZE - BORDER - radius, size=2)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(150, 120, 60))
    return img


def make_cube_mesh(size: float, texture: Image.Image) -> trimesh.Trimesh:
    h = size / 2
    # 各面4頂点（法線方向, 面内のu軸, v軸）
    faces_def = [
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    ]
    edge = BORDER / 2 / TEX_SIZE  # 縁の中のUV座標
    vertices, faces, uvs = [], [], []
    for i, (n, u, v) in enumerate(faces_def):
        n, u, v = (np.array(a, dtype=float) for a in (n, u, v))
        corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for su, sv in corners:
            vertices.append(h * (n + su * u + sv * v))
            if i == 0:
                uvs.append(((su + 1) / 2, (sv + 1) / 2))
            else:
                uvs.append((edge, edge))
        b = 4 * i
        faces += [(b, b + 1, b + 2), (b, b + 2, b + 3)]
    visual = trimesh.visual.TextureVisuals(
        uv=np.array(uvs), material=trimesh.visual.material.SimpleMaterial(image=texture)
    )
    return trimesh.Trimesh(np.array(vertices), np.array(faces), visual=visual, process=False)


def export_part(path: Path, size: float, defect: Defect, rng: np.random.Generator) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = make_cube_mesh(size, make_texture(defect, rng))
    # glbはY-upの規格で、Genesisは読み込み時に (X, Y, Z) -> (X, -Z, Y) と変換する。
    # 逆変換 (x, y, z) -> (x, z, -y) をかけておき、Z-upのまま読み込まれるようにする
    mesh.vertices = mesh.vertices[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
    mesh.export(path)
    return path
