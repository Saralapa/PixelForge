#!/usr/bin/env python3
"""Baixa os pesos do modelo configurado uma única vez, para uso 100% offline
depois disso.

Uso:
    python scripts/download_model.py

Depois de rodar este script, a aplicação principal (run.py) nunca mais
precisa de rede: ela carrega os pesos do cache local em ./models e roda
com HF_HUB_OFFLINE=1.

Se você já tem um checkpoint .safetensors baixado manualmente (ex.: do
Civitai), não precisa deste script — só aponte config.json para o arquivo,
com "model_source": "file".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODELS_DIR, load_settings  # noqa: E402


def main() -> None:
    settings = load_settings()

    if settings.model_source == "file":
        print(
            "config.json está configurado com model_source='file' — "
            "nenhum download é necessário. Verifique se o arquivo em "
            f"'{settings.model_file}' existe."
        )
        return

    if settings.pipeline == "sdxl":
        from diffusers import StableDiffusionXLPipeline as PipelineClass
    else:
        from diffusers import StableDiffusionPipeline as PipelineClass

    print(f"Baixando '{settings.model_id}' para {MODELS_DIR} ...")
    print("Isso só acontece uma vez. Pode levar alguns minutos.")

    PipelineClass.from_pretrained(
        settings.model_id,
        cache_dir=str(MODELS_DIR),
        safety_checker=None,
        requires_safety_checker=False,
    )

    print("Download concluído. A partir de agora, a aplicação roda 100% offline.")


if __name__ == "__main__":
    main()
