"""Claudeによる検査：LLMが基準書を判定ルールに変換し、VLMが画像とルールから良否を判定する。"""

import base64
import io
import json
from typing import Literal

import anthropic
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

DefectType = Literal["none", "scratch", "stain", "chip", "other"]


class Rule(BaseModel):
    defect_type: Literal["scratch", "stain", "chip"]
    ng_condition: str = Field(description="NGとする条件（日本語、1文）")
    threshold_ratio: float = Field(
        description="NGになる下限の比率。傷は長さ/一辺、色ムラは面積/上面。少しでもあればNGなら0"
    )


class Rules(BaseModel):
    rules: list[Rule]


class Judgment(BaseModel):
    verdict: Literal["OK", "NG"]
    defect_type: DefectType
    estimated_ratio: float = Field(description="見つけた不良の大きさの比率。なければ0")
    confidence: float = Field(description="0から1")
    reason: str = Field(description="判定理由（日本語、1〜2文）")


RULES_SYSTEM = (
    "あなたは製造ラインの品質管理担当者です。検査基準書を読み、外観検査の判定ルールに変換してください。"
    "基準書に書かれていない条件を追加してはいけません。"
)

JUDGE_SYSTEM = (
    "あなたは外観検査員です。部品の上面を真上から撮影した画像と判定ルールが与えられます。"
    "画像の中央の灰色の正方形が部品の上面です。部品の外（背景や影）は検査対象ではありません。"
    "傷は細い黒い線、色ムラは茶色く変色した部分です。"
    "不良を探し、その大きさを部品の一辺または上面の面積に対する比率で見積もり、ルールに照らしてOKかNGかを判定してください。"
)


class Inspector:
    def __init__(self, model: str, effort: str = "medium"):
        load_dotenv()
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def _parse(self, system: str, content: list, output_format: type[BaseModel]):
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_format=output_format,
            output_config={"effort": self.effort},
            # 安全分類器に誤って拒否された場合、サーバー側で別モデルに切り替えて再実行する
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage["input_tokens"] += response.usage.input_tokens
        self.usage["output_tokens"] += response.usage.output_tokens
        if response.stop_reason == "refusal":
            raise RuntimeError(f"refused: {response.stop_details}")
        return response.parsed_output

    def read_standard(self, standard_text: str) -> Rules:
        """LLM：検査基準書を判定ルール（JSON）に変換する。"""
        return self._parse(RULES_SYSTEM, [{"type": "text", "text": standard_text}], Rules)

    def judge(self, image: Image.Image, rules: Rules) -> Judgment:
        """VLM：部品の画像とルールから良否を判定する。"""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(buf.getvalue()).decode("utf-8"),
                },
            },
            {
                "type": "text",
                "text": "判定ルール:\n" + json.dumps(rules.model_dump(), ensure_ascii=False, indent=2),
            },
        ]
        return self._parse(JUDGE_SYSTEM, content, Judgment)
