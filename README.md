# PixelForge

Gerador de imagens por IA a partir de texto, rodando **inteiramente na sua
máquina**. Sem contas, sem chaves de API, sem servidor na nuvem, sem
telemetria e sem qualquer filtro de conteúdo — o prompt que você digita é o
prompt que o modelo recebe.

## O que é e o que não é

- **É:** uma interface web local (FastAPI + HTML/CSS/JS puro) na frente de um
  pipeline Stable Diffusion (via [diffusers](https://github.com/huggingface/diffusers))
  rodando na sua GPU.
- **Não é:** um cliente para nenhuma API de terceiros. Depois do download
  inicial dos pesos do modelo, a aplicação roda com `HF_HUB_OFFLINE=1` e o
  servidor escuta só em `127.0.0.1` — nada sai da máquina.
- **Não tem:** `safety_checker`, blocklist de palavras, verificação de
  política de uso ou qualquer outra trava. Ver [Ausência de filtros](#ausência-de-filtros-e-o-que-isso-significa-na-prática).

## Requisitos

- Python 3.10+
- GPU NVIDIA com CUDA (recomendado) — testado como alvo para uma RTX 2080
  Super (8 GB de VRAM). Também funciona em Mac Apple Silicon (MPS) ou só CPU,
  mais lentamente; veja [Hardware e desempenho](#hardware-e-desempenho).
- ~6 GB de espaço em disco para os pesos do Stable Diffusion 1.5.

## Instalação

```bash
git clone <url-deste-repositorio>
cd pixelforge

# 1. Ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. PyTorch — escolha o comando certo para SUA GPU em
#    https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Resto das dependências
pip install -r requirements.txt

# 4. Baixe os pesos uma única vez (precisa de internet só nesta etapa)
python scripts/download_model.py

# 5. Rode
python run.py
```

O navegador abre sozinho em `http://127.0.0.1:7860`. Nas próximas vezes,
`start.sh` (Linux/macOS) ou `start.bat` (Windows) fazem os passos 1–5 de
novo automaticamente, pulando o que já está pronto.

### Já tenho um checkpoint `.safetensors` baixado

Não precisa rodar `download_model.py`. Edite `config.json`:

```json
{
  "model_source": "file",
  "model_file": "models/meu-modelo.safetensors"
}
```

## Usando

1. Digite o prompt na caixa de texto.
2. Clique em **Gerar** (ou `Ctrl+Enter`).
3. Acompanhe a barra de progresso passo a passo.
4. A imagem aparece com um botão de download e a seed usada — clique na
   seed para reaproveitá-la numa próxima geração.

O painel **Avançado** (recolhido por padrão) expõe prompt negativo, número
de passos, CFG (guidance scale), largura, altura, seed e quantidade de
imagens por geração.

## Hardware e desempenho

O `config.json` já vem configurado para uma RTX 2080 Super: Stable Diffusion
1.5, 512×512, 25 passos, fp16, agendador DPM++ 2M Karras — cerca de 2 a 3
segundos por imagem, com folga nos 8 GB de VRAM.

| Hardware | Configuração sugerida | Tempo aproximado por imagem |
|---|---|---|
| GPU NVIDIA 8GB+ (ex.: RTX 2080 Super) | `sd15`, 512×512, fp16 (padrão) | 2–5 s |
| GPU NVIDIA 12GB+ | `sdxl`, 1024×1024 | 15–30 s |
| Mac Apple Silicon (M1+) | `sd15`, 512×512 (detecta MPS sozinho) | 20–60 s |
| Só CPU | `sd15`, 512×512, poucos passos | 30–90 s+ |

Para usar SDXL, em `config.json`:

```json
{
  "pipeline": "sdxl",
  "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
  "defaults": { "width": 1024, "height": 1024, "steps": 30 }
}
```

O código detecta `cuda` → `mps` → `cpu` sozinho e ajusta o tipo de dado
(`float16` em GPU, `float32` em CPU) — não precisa configurar nada disso à
mão.

## Ausência de filtros e o que isso significa na prática

Por pedido explícito no design deste projeto, não há nenhuma camada de
moderação:

- O carregamento do pipeline usa `safety_checker=None` e
  `requires_safety_checker=False` — o classificador NSFW do Stable Diffusion
  **nem chega a ser instanciado**, então nenhuma imagem sai borrada ou preta.
- O prompt do `<textarea>` vai direto para o tokenizer do modelo: não há
  blocklist, regex, checagem de palavra-chave ou reescrita.
- Não existe nenhum classificador, watermark ou pós-processamento entre o
  VAE e o arquivo PNG salvo em disco.
- Não há telemetria nem chamada de rede durante a geração.

Isso é uma escolha de código, não de licença: os pesos do Stable Diffusion
são distribuídos sob a licença **CreativeML OpenRAIL-M**, cujos termos de
uso (incluindo restrições sobre certos tipos de conteúdo) acompanham o
modelo e continuam valendo independentemente do que este software permite
tecnicamente. Use por sua conta e responsabilidade.

## Arquitetura

```
pixelforge/
├── app/
│   ├── config.py   # leitura de config.json + variáveis de ambiente
│   ├── engine.py   # carga do pipeline, geração, sem filtros
│   ├── jobs.py     # fila de jobs em memória + progresso por passo
│   └── main.py     # rotas FastAPI
├── web/            # interface estática (HTML/CSS/JS puro, sem CDN)
├── scripts/
│   └── download_model.py
├── models/         # pesos baixados (não versionado)
├── outputs/        # imagens geradas (não versionado)
├── config.json
├── run.py          # ponto de entrada
├── start.sh / start.bat
└── requirements.txt
```

### API interna

| Rota | Descrição |
|---|---|
| `GET /` | interface |
| `GET /api/status` | dispositivo detectado, modelo, se já está carregado |
| `POST /api/generate` | inicia uma geração, devolve `{job_id}` |
| `GET /api/jobs/{id}` | status/progresso/resultado do job |
| `GET /api/image/{nome}` | serve o PNG gerado |

## Privacidade

O servidor escuta apenas em `127.0.0.1` (configurável em `config.json`, mas
o padrão nunca expõe a aplicação na rede). Prompts e imagens ficam só no seu
disco, em `outputs/`; nada é enviado a lugar nenhum.
