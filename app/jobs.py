"""Registro de jobs de geração em memória.

Cada chamada a /api/generate cria um job e devolve seu id imediatamente; a
geração roda numa thread de trabalho para não bloquear o event loop do
FastAPI, e o cliente acompanha o progresso via polling em /api/jobs/{id}.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from . import engine
from .config import Settings

# Uma única worker thread: a GPU é serializada de qualquer forma pelo lock
# do pipeline em engine.py, então não há ganho em paralelizar aqui — só
# queremos tirar a geração do event loop principal.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pixelforge-gen")
_JOBS: dict[str, "Job"] = {}
_JOBS_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued -> running -> done | error
    step: int = 0
    total_steps: int = 0
    images: list[str] = field(default_factory=list)
    seed: Optional[int] = None
    error: Optional[str] = None
    elapsed_seconds: Optional[float] = None


def get_job(job_id: str) -> Optional[Job]:
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def _run(job_id: str, settings: Settings, request: "engine.GenerationRequest") -> None:
    job = _JOBS[job_id]
    job.status = "running"
    job.total_steps = request.steps

    def on_step(step: int, total: int) -> None:
        job.step = step
        job.total_steps = total

    try:
        result = engine.generate(settings, request, on_step=on_step)
        job.images = [p.name for p in result.images]
        job.seed = result.seed
        job.elapsed_seconds = result.elapsed_seconds
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 — reportado ao cliente via API
        job.error = str(exc)
        job.status = "error"


def submit(settings: Settings, request: "engine.GenerationRequest") -> str:
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, total_steps=request.steps)
    with _JOBS_LOCK:
        _JOBS[job_id] = job
    _EXECUTOR.submit(_run, job_id, settings, request)
    return job_id
