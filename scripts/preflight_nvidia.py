"""실제 NVIDIA 모델 카탈로그에서 수업 모델의 현재 상태를 확인한다."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from verifiable_ai_workflow.config import load_project_env, load_settings, require_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_URL = "https://integrate.api.nvidia.com/v1/models"
CATALOG_MODELS = (
    (
        "multimodal",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "English only",
    ),
    ("multimodal", "nvidia/nemotron-nano-12b-v2-vl", "English only"),
    (
        "multimodal",
        "meta/llama-3.2-11b-vision-instruct",
        "image+text: English only",
    ),
    (
        "multimodal",
        "google/diffusiongemma-26b-a4b-it",
        "multilingual",
    ),
    ("multimodal", "google/gemma-4-31b-it", "35+ languages; pre-trained on 140+"),
    (
        "multimodal",
        "minimaxai/minimax-m3",
        "official list not specified",
    ),
    (
        "multimodal",
        "stepfun-ai/step-3.7-flash",
        "official list not specified",
    ),
    (
        "multimodal",
        "meta/llama-3.2-90b-vision-instruct",
        "image+text: English only",
    ),
    ("multimodal", "moonshotai/kimi-k2.6", "official list not specified"),
    ("text", "openai/gpt-oss-20b", "official list not specified"),
    ("text", "deepseek-ai/deepseek-v4-flash", "official list not specified"),
    ("text", "deepseek-ai/deepseek-v4-pro", "official list not specified"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA NIM 모델 목록 사전 확인")
    parser.add_argument("--config", default="configs/nvidia-nim.yaml")
    args = parser.parse_args()
    load_project_env(PROJECT_ROOT)
    settings = load_settings(PROJECT_ROOT / args.config)
    if not settings.provider.api_key_env:
        raise ValueError("NVIDIA NIM 설정에 api_key_env가 필요합니다")
    api_key = require_api_key(settings.provider.api_key_env)
    request = urllib.request.Request(
        CATALOG_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        catalog = json.load(response)

    available = {item["id"] for item in catalog.get("data", [])}
    configured = settings.provider.model.removeprefix("nvidia_nim/")
    print(f"configured model: {configured}")
    print(f"available now: {configured in available}")
    print("catalog models:")
    for category, model, languages in CATALOG_MODELS:
        print(f"- [{category}] {model}: {model in available}; languages={languages}")
    return 0 if configured in available else 1


if __name__ == "__main__":
    raise SystemExit(main())
