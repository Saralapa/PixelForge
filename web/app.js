// PixelForge — lógica da interface. Sem framework, sem dependências
// externas: só fetch() contra a própria API local e polling de progresso.
(() => {
  const $ = (id) => document.getElementById(id);

  const promptEl = $("prompt");
  const negativeEl = $("negative-prompt");
  const widthEl = $("width");
  const heightEl = $("height");
  const stepsEl = $("steps");
  const guidanceEl = $("guidance");
  const numImagesEl = $("num-images");
  const seedEl = $("seed");

  const generateBtn = $("generate-btn");
  const progressWrap = $("progress-wrap");
  const progressFill = $("progress-fill");
  const progressLabel = $("progress-label");
  const errorBox = $("error-box");
  const gallery = $("gallery");
  const statusLine = $("status-line");

  let polling = null;

  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
  }

  function clearError() {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
  }

  function setBusy(busy) {
    generateBtn.disabled = busy;
    generateBtn.textContent = busy ? "Gerando…" : "Gerar";
  }

  function setProgress(step, total) {
    progressWrap.classList.remove("hidden");
    const pct = total > 0 ? Math.round((step / total) * 100) : 0;
    progressFill.style.width = pct + "%";
    progressLabel.textContent = `Gerando… passo ${step}/${total}`;
  }

  function hideProgress() {
    progressWrap.classList.add("hidden");
    progressFill.style.width = "0%";
  }

  function renderImages(images, seed) {
    gallery.innerHTML = "";
    for (const name of images) {
      const card = document.createElement("div");
      card.className = "image-card";

      const img = document.createElement("img");
      img.src = `/api/image/${encodeURIComponent(name)}`;
      img.alt = "Imagem gerada";
      card.appendChild(img);

      const footer = document.createElement("div");
      footer.className = "image-card-footer";

      const seedBtn = document.createElement("button");
      seedBtn.textContent = `seed ${seed}`;
      seedBtn.title = "Reutilizar esta seed";
      seedBtn.addEventListener("click", () => {
        seedEl.value = seed;
      });
      footer.appendChild(seedBtn);

      const downloadLink = document.createElement("a");
      downloadLink.href = `/api/image/${encodeURIComponent(name)}`;
      downloadLink.download = name;
      downloadLink.textContent = "baixar";
      footer.appendChild(downloadLink);

      card.appendChild(footer);
      gallery.appendChild(card);
    }
  }

  async function loadStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      const model = data.model_id || "?";
      statusLine.textContent = `Dispositivo: ${data.device} · Modelo: ${model}${
        data.model_loaded ? "" : " (carrega na primeira geração)"
      }`;
    } catch (e) {
      statusLine.textContent = "Não foi possível consultar o status do servidor.";
    }
  }

  function buildPayload() {
    return {
      prompt: promptEl.value.trim(),
      negative_prompt: negativeEl.value.trim(),
      width: parseInt(widthEl.value, 10),
      height: parseInt(heightEl.value, 10),
      steps: parseInt(stepsEl.value, 10),
      guidance_scale: parseFloat(guidanceEl.value),
      num_images: parseInt(numImagesEl.value, 10),
      seed: parseInt(seedEl.value, 10),
    };
  }

  async function pollJob(jobId) {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();

      if (job.status === "running" || job.status === "queued") {
        setProgress(job.step, job.total_steps || 1);
        polling = setTimeout(() => pollJob(jobId), 400);
        return;
      }

      if (job.status === "error") {
        hideProgress();
        setBusy(false);
        showError(job.error || "Falha desconhecida na geração.");
        return;
      }

      // done
      hideProgress();
      setBusy(false);
      renderImages(job.images, job.seed);
      loadStatus();
    } catch (e) {
      hideProgress();
      setBusy(false);
      showError("Perdi a conexão com o servidor durante a geração.");
    }
  }

  async function generate() {
    const payload = buildPayload();
    if (!payload.prompt) {
      showError("Digite um prompt antes de gerar.");
      return;
    }

    clearError();
    setBusy(true);
    setProgress(0, payload.steps);
    if (polling) clearTimeout(polling);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Erro HTTP ${res.status}`);
      }
      const { job_id } = await res.json();
      pollJob(job_id);
    } catch (e) {
      hideProgress();
      setBusy(false);
      showError(e.message || "Falha ao iniciar a geração.");
    }
  }

  generateBtn.addEventListener("click", generate);
  promptEl.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      generate();
    }
  });

  loadStatus();
})();
