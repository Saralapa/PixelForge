"""Carregamento de configuração do PixelForge.

Lê config.json na raiz do projeto. Chaves começadas com "_" são apenas
comentários e são ignoradas. Variáveis de ambiente com prefixo PIXELFORGE_
sobrescrevem os campos de topo (ex.: PIXELFORGE_PORT=8000).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
WEB_DIR = ROOT_DIR / "web"
CONFIG_PATH = ROOT_DIR / "config.json"


@dataclass
class GenerationDefaults:
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: int = -1


@dataclass
class Settings:
    pipeline: str = "sd15"
    model_source: str = "hf"
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    model_file: str = "models/meu-modelo.safetensors"
    host: str = "127.0.0.1"
    port: int = 7860
    defaults: GenerationDefaults = field(default_factory=GenerationDefaults)

    @property
    def resolved_model_file(self) -> Path:
        p = Path(self.model_file)
        return p if p.is_absolute() else ROOT_DIR / p


def _strip_comments(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_settings() -> Settings:
    raw: dict = {}
    if CONFIG_PATH.exists():
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = _strip_comments(raw)

    defaults_raw = _strip_comments(raw.pop("defaults", {}) or {})
    defaults = GenerationDefaults(**{
        k: v for k, v in defaults_raw.items() if k in GenerationDefaults.__dataclass_fields__
    })

    settings = Settings(
        **{k: v for k, v in raw.items() if k in Settings.__dataclass_fields__ and k != "defaults"},
        defaults=defaults,
    )

    # Overrides por variável de ambiente (útil para start.sh / start.bat / testes).
    if v := os.environ.get("PIXELFORGE_HOST"):
        settings.host = v
    if v := os.environ.get("PIXELFORGE_PORT"):
        settings.port = int(v)
    if v := os.environ.get("PIXELFORGE_MODEL_ID"):
        settings.model_id = v
    if v := os.environ.get("PIXELFORGE_MODEL_SOURCE"):
        settings.model_source = v
    if v := os.environ.get("PIXELFORGE_PIPELINE"):
        settings.pipeline = v

    return settings


MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
