"""Motor de geração de imagens do PixelForge.

Carrega um pipeline do diffusers (Stable Diffusion 1.5 ou SDXL) inteiramente
a partir do disco local, sem qualquer camada de moderação de conteúdo:

  * `safety_checker=None` e `requires_safety_checker=False` na carga — o
    classificador NSFW do Stable Diffusion nunca é instanciado.
  * O prompt do usuário vai direto para o tokenizer: não há blocklist,
    regex, checagem de palavra-chave ou reescrita de nenhum tipo.
  * Nenhum pós-processamento, watermark ou classificador roda entre o VAE
    e o arquivo PNG salvo em disco.
  * Nenhuma chamada de rede acontece durante a geração (HF_HUB_OFFLINE=1
    é definido em app/main.py antes de qualquer import do diffusers).

Os pesos do Stable Diffusion são licenciados sob CreativeML OpenRAIL-M; os
termos dessa licença acompanham o modelo e são responsabilidade de quem o
usa, independentemente deste código.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .config import MODELS_DIR, OUTPUTS_DIR, Settings

# ---------------------------------------------------------------------------
# Estado do módulo: pipeline carregado preguiçosamente, protegido por lock.
# Uma GPU só faz uma geração por vez; o lock também protege a carga inicial.
# ---------------------------------------------------------------------------
_PIPELINE = None
_PIPELINE_LOCK = threading.Lock()
_DEVICE: Optional[str] = None
_DTYPE: Optional[torch.dtype] = None
_LOADED_MODEL_NAME: Optional[str] = None


@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 25
    guidance_scale: float = 7.5
    num_images: int = 1
    seed: int = -1


@dataclass
class GenerationResult:
    images: list[Path]
    seed: int
    elapsed_seconds: float


def detect_device() -> str:
    """cuda -> mps -> cpu, nessa ordem de preferência."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device: str) -> torch.dtype:
    # A RTX 2080 Super (Turing) não tem núcleos bf16 nativos; fp16 é o
    # formato certo para CUDA. CPU precisa de fp32 (fp16 não é bem
    # suportado nas operações de CPU do PyTorch).
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = detect_device()
    return _DEVICE


def is_loaded() -> bool:
    return _PIPELINE is not None


def loaded_model_name() -> Optional[str]:
    return _LOADED_MODEL_NAME


def _apply_memory_optimizations(pipe, device: str, pipeline_kind: str) -> None:
    # Reduz o pico de VRAM fatiando a atenção e o decode do VAE — mantém
    # tudo confortável em placas de 8 GB como a RTX 2080 Super.
    pipe.enable_attention_slicing()
    # pipe.enable_vae_slicing() foi removido nas versões novas do diffusers;
    # o jeito atual é chamar direto no VAE. Mantém o fallback para as antigas.
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    elif hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if pipeline_kind == "sdxl" and device == "cuda":
        # SDXL em 1024x1024 é pesado demais para caber tranquilo em 8 GB;
        # offload move módulos para a CPU quando não estão em uso.
        pipe.enable_model_cpu_offload()
    elif device == "cuda":
        pipe.to(device)
    else:
        pipe.to(device)


def _configure_scheduler(pipe) -> None:
    from diffusers import DPMSolverMultistepScheduler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )


def load_pipeline(settings: Settings):
    """Carrega (ou devolve) o pipeline, preguiçosamente e de forma segura
    entre threads."""
    global _PIPELINE, _LOADED_MODEL_NAME

    with _PIPELINE_LOCK:
        if _PIPELINE is not None:
            return _PIPELINE

        device = get_device()
        dtype = dtype_for_device(device)
        global _DTYPE
        _DTYPE = dtype

        if settings.pipeline == "sdxl":
            from diffusers import StableDiffusionXLPipeline as PipelineClass
        else:
            from diffusers import StableDiffusionPipeline as PipelineClass

        common_kwargs = dict(
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        if settings.model_source == "file":
            model_path = settings.resolved_model_file
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Arquivo de modelo não encontrado: {model_path}. "
                    "Ajuste 'model_file' em config.json."
                )
            pipe = PipelineClass.from_single_file(str(model_path), **common_kwargs)
            _LOADED_MODEL_NAME = model_path.name
        else:
            pipe = PipelineClass.from_pretrained(
                settings.model_id,
                cache_dir=str(MODELS_DIR),
                **common_kwargs,
            )
            _LOADED_MODEL_NAME = settings.model_id

        _configure_scheduler(pipe)
        _apply_memory_optimizations(pipe, device, settings.pipeline)
        pipe.set_progress_bar_config(disable=True)

        _PIPELINE = pipe
        return _PIPELINE


def generate(
    settings: Settings,
    request: GenerationRequest,
    on_step: Optional[Callable[[int, int], None]] = None,
) -> GenerationResult:
    """Gera `request.num_images` imagens de forma síncrona. Deve rodar em
    thread de trabalho — é uma chamada bloqueante e serializada pelo lock
    do pipeline (uma GPU processa uma geração por vez)."""
    pipe = load_pipeline(settings)
    device = get_device()

    seed = request.seed
    if seed is None or seed < 0:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
    generator = torch.Generator(device=device if device != "mps" else "cpu").manual_seed(seed)

    total_steps = request.steps

    def _callback(pipe_, step_index, timestep, callback_kwargs):
        if on_step is not None:
            on_step(step_index + 1, total_steps)
        return callback_kwargs

    with _PIPELINE_LOCK:
        start = time.monotonic()
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            num_images_per_prompt=request.num_images,
            generator=generator,
            callback_on_step_end=_callback,
        )
        elapsed = time.monotonic() - start

    saved_paths: list[Path] = []
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i, image in enumerate(result.images):
        meta = PngInfo()
        meta.add_text("prompt", request.prompt)
        meta.add_text("negative_prompt", request.negative_prompt or "")
        meta.add_text("steps", str(request.steps))
        meta.add_text("guidance_scale", str(request.guidance_scale))
        meta.add_text("seed", str(seed))
        meta.add_text("model", _LOADED_MODEL_NAME or "")
        filename = f"{stamp}-{seed}-{i}.png" if request.num_images > 1 else f"{stamp}-{seed}.png"
        out_path = OUTPUTS_DIR / filename
        image.save(out_path, pnginfo=meta)
        saved_paths.append(out_path)

    return GenerationResult(images=saved_paths, seed=seed, elapsed_seconds=elapsed)
