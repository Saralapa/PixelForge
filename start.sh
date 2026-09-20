#!/usr/bin/env bash
# PixelForge — inicialização em Linux/macOS.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Criando ambiente virtual em .venv ..."
  python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import torch" >/dev/null 2>&1; then
  echo
  echo "PyTorch não está instalado neste ambiente."
  echo "Instale a versão certa para sua GPU antes de continuar, por exemplo:"
  echo "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
  echo
  exit 1
fi

pip install -q -r requirements.txt

if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
  echo "Nenhum modelo em cache encontrado. Baixando na primeira vez..."
  python scripts/download_model.py
fi

python run.py
