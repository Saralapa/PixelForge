#!/usr/bin/env python3
"""Ponto de entrada do PixelForge: sobe o servidor local e abre o navegador.

Uso:
    python run.py
"""
from __future__ import annotations

import os
import threading
import webbrowser

# Definido antes de importar app.main, para garantir que nenhuma biblioteca
# tente falar com a rede em nenhum momento da inicialização.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")

import uvicorn

from app.config import load_settings


def _open_browser_when_ready(url: str) -> None:
    import time
    import urllib.request

    for _ in range(120):  # até 60s esperando o servidor subir
        try:
            urllib.request.urlopen(url, timeout=0.5)
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(0.5)


def main() -> None:
    settings = load_settings()
    url = f"http://{settings.host}:{settings.port}/"

    print("=" * 60)
    print(" PixelForge — gerador local de imagens por IA")
    print("=" * 60)
    print(f" Interface: {url}")
    print(" Execução 100% local — nenhuma chamada sai desta máquina.")
    print(" Pressione Ctrl+C para encerrar.")
    print("=" * 60)

    threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
