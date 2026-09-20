@echo off
REM PixelForge — inicialização em Windows.
cd /d "%~dp0"

if not exist ".venv" (
    echo Criando ambiente virtual em .venv ...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo.
    echo PyTorch nao esta instalado neste ambiente.
    echo Instale a versao certa para sua GPU antes de continuar, por exemplo:
    echo   pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo.
    exit /b 1
)

pip install -q -r requirements.txt

if not exist "models\*" (
    echo Nenhum modelo em cache encontrado. Baixando na primeira vez...
    python scripts\download_model.py
)

python run.py
