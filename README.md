# AI検品アーム

検査基準書をLLMが読み、VLMが部品の良否を判定し、シミュレーション上のFranka Pandaが良品箱とNG箱に仕分ける。

- シミュレータ：Genesis
- ロボット：Franka Panda
- LLM・VLM：Claude API

## セットアップ

```bash
uv sync                    # Step1 に必要なもの
uv sync --extra train      # Step2（LeRobot）も入れる場合
cp .env.example .env       # ANTHROPIC_API_KEY を記入
uv run python scripts/check_env.py
```

