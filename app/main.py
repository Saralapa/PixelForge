"""PixelForge — servidor FastAPI.

Serve a interface estática e uma API mínima para gerar imagens localmente.
Não há nenhuma rota de moderação, filtro ou verificação de política: o
prompt recebido em /api/generate vai direto para o motor em engine.py.
"""
from __future__ import annotations

import os

# Precisa ser definido antes de qualquer import que puxe transformers /
# huggingface_hub (engine.py os importa preguiçosamente, mas garantimos
# aqui também, no topo do processo). Depois do primeiro download feito por
# scripts/download_model.py, isso impede qualquer chamada de rede.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import engine, jobs
from .config import OUTPUTS_DIR, WEB_DIR, load_settings

app = FastAPI(title="PixelForge", docs_url=None, redoc_url=None)

settings = load_settings()


class GenerateBody(BaseModel):
    prompt: str = Field(..., min_length=1)
    negative_prompt: str = ""
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    steps: int = Field(25, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=0, le=30)
    num_images: int = Field(1, ge=1, le=4)
    seed: int = -1


@app.get("/api/status")
def status():
    return {
        "device": engine.get_device(),
        "pipeline": settings.pipeline,
        "model_source": settings.model_source,
        "model_id": settings.model_id if settings.model_source == "hf" else settings.model_file,
        "model_loaded": engine.is_loaded(),
        "loaded_model_name": engine.loaded_model_name(),
        "defaults": settings.defaults.__dict__,
    }


@app.post("/api/generate")
def generate_endpoint(body: GenerateBody):
    request = engine.GenerationRequest(
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        guidance_scale=body.guidance_scale,
        num_images=body.num_images,
        seed=body.seed,
    )
    job_id = jobs.submit(settings, request)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job não encontrado")
    return {
        "id": job.id,
        "status": job.status,
        "step": job.step,
        "total_steps": job.total_steps,
        "images": job.images,
        "seed": job.seed,
        "error": job.error,
        "elapsed_seconds": job.elapsed_seconds,
    }


@app.get("/api/image/{name}")
def get_image(name: str):
    # Impede path traversal: só serve arquivos que ficam de fato dentro de
    # OUTPUTS_DIR, pelo nome de arquivo puro.
    safe_name = Path(name).name
    path = (OUTPUTS_DIR / safe_name).resolve()
    if OUTPUTS_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="imagem não encontrada")
    return FileResponse(path)


# Interface estática. Montado por último para não conflitar com /api/*.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
