// Models panel — scans project models/ directory and displays model cards
// Calls window.artifex.scanModels() and window.artifex.deleteModel(path) via IPC bridge

let modelsGrid = null;
let modelsLoading = false;

async function initModelsPanel(container) {
  container.innerHTML = '';

  // Inject panel-specific styles once
  injectModelsPanelStyles();

  // Header
  const header = document.createElement('div');
  header.className = 'panel-header';
  header.innerHTML = `
    <h2>Models</h2>
    <div>
      <button class="btn btn-primary btn-sm" id="models-refresh-btn">Refresh</button>
    </div>
  `;
  container.appendChild(header);

  // Grid container
  modelsGrid = document.createElement('div');
  modelsGrid.className = 'card-grid models-grid';
  container.appendChild(modelsGrid);

  // Wire up refresh
  document.getElementById('models-refresh-btn').onclick = () => loadModels();

  // Initial load
  await loadModels();
}

function cleanupModelsPanel() {
  modelsGrid = null;
  modelsLoading = false;
}

async function loadModels() {
  if (modelsLoading || !modelsGrid) return;
  modelsLoading = true;

  modelsGrid.innerHTML = `
    <div class="models-loading">Scanning Transformers and Ollama models...</div>
  `;

  // Query all three backends in parallel — Ollama may not be running and
  // llama.cpp may have no config; both resolve to [] on miss rather than
  // throwing, so a single broken backend doesn't blank the whole grid.
  const [transformersResult, ollamaResult, llamaCppResult] = await Promise.allSettled([
    window.artifex.scanModels(),
    window.artifex.scanOllamaModels ? window.artifex.scanOllamaModels() : Promise.resolve([]),
    window.artifex.scanLlamaCppModels ? window.artifex.scanLlamaCppModels() : Promise.resolve([]),
  ]);

  modelsGrid.innerHTML = '';

  const transformerModels = transformersResult.status === 'fulfilled'
    ? (transformersResult.value || [])
    : [];
  const ollamaModels = ollamaResult.status === 'fulfilled'
    ? (ollamaResult.value || [])
    : [];
  const llamaCppModels = llamaCppResult.status === 'fulfilled'
    ? (llamaCppResult.value || [])
    : [];

  if (transformersResult.status === 'rejected') {
    const err = transformersResult.reason;
    modelsGrid.appendChild(createScanErrorRow(
      `Failed to scan Transformers models: ${err?.message || err}`
    ));
  }
  if (llamaCppResult.status === 'rejected') {
    const err = llamaCppResult.reason;
    modelsGrid.appendChild(createScanErrorRow(
      `Failed to scan llama.cpp models: ${err?.message || err}`
    ));
  }

  if (transformerModels.length === 0 && ollamaModels.length === 0 && llamaCppModels.length === 0) {
    modelsGrid.innerHTML = `
      <div class="models-empty">
        No models found. Drop Transformers checkpoints in the models/ directory,
        pull Ollama models with <code>ollama pull &lt;name&gt;</code>, or declare
        llama.cpp models in <code>llama_cpp_config.json</code>.
      </div>
    `;
    modelsLoading = false;
    return;
  }

  for (const model of transformerModels) {
    modelsGrid.appendChild(createModelCard(model));
  }
  for (const model of ollamaModels) {
    modelsGrid.appendChild(createOllamaModelCard(model));
  }
  for (const model of llamaCppModels) {
    modelsGrid.appendChild(createLlamaCppModelCard(model));
  }

  modelsLoading = false;
}

function createScanErrorRow(message) {
  const row = document.createElement('div');
  row.className = 'models-empty';
  row.style.color = 'var(--error)';
  row.textContent = message;
  return row;
}

function createModelCard(model) {
  const card = document.createElement('div');
  card.className = 'model-card';
  card.dataset.path = model.path;
  card.dataset.source = 'transformers';

  // Badge class based on quant type
  let badgeClass = 'badge-bf16';
  if (model.isQuantized) {
    badgeClass = model.mixedPrecision ? 'badge-mixed' : 'badge-quant';
  }

  // Format vocab nicely (e.g., 248320 -> "248K")
  const vocabStr = model.vocabSize >= 1000
    ? (model.vocabSize / 1000).toFixed(model.vocabSize % 1000 === 0 ? 0 : 1) + 'K'
    : String(model.vocabSize);

  card.innerHTML = `
    <div class="model-card-header">
      <div class="model-name">${escapeHtml(model.name)}</div>
      <span class="model-badge ${badgeClass}">${escapeHtml(model.quantDetail)}</span>
    </div>
    <div class="model-source-row">
      <span class="model-source-badge source-transformers">Transformers</span>
    </div>
    <div class="model-size">${escapeHtml(model.sizeFormatted)}</div>
    <div class="model-meta-grid">
      <div class="model-meta-item">
        <span class="model-meta-label">Type</span>
        <span class="model-meta-value">${escapeHtml(model.modelType)}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Layers</span>
        <span class="model-meta-value">${model.numLayers}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Hidden</span>
        <span class="model-meta-value">${model.hiddenSize.toLocaleString()}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Vocab</span>
        <span class="model-meta-value">${vocabStr}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Shards</span>
        <span class="model-meta-value">${model.shardCount}</span>
      </div>
      ${model.isQuantized && model.groupSize ? `
      <div class="model-meta-item">
        <span class="model-meta-label">Group</span>
        <span class="model-meta-value">${model.groupSize}</span>
      </div>` : ''}
    </div>
    <div class="model-cache-row"></div>
    <div class="model-card-actions"></div>
  `;

  // NF4 cache row + skip-cache toggle. Transformers-only feature: the
  // engine writes a sibling `<name>-nf4-cached/` directory to skip
  // re-quantizing on subsequent loads. Users who'd rather keep the disk
  // can flip this off — it writes a `.no_cache` marker to the model
  // directory and deletes the existing cache. The Python engine respects
  // the marker on future loads.
  const cacheRow = card.querySelector('.model-cache-row');
  cacheRow.appendChild(buildCacheControl(model, card));

  // Delete button
  const actions = card.querySelector('.model-card-actions');
  const deleteBtn = document.createElement('button');
  deleteBtn.className = 'btn btn-danger btn-sm';
  deleteBtn.textContent = 'Delete';
  deleteBtn.onclick = async (e) => {
    e.stopPropagation();
    const confirmed = await showModal(
      'Delete Model',
      `Permanently delete "${model.name}"? This will remove ${model.sizeFormatted} from disk. This action cannot be undone.`
    );
    if (confirmed) {
      deleteBtn.disabled = true;
      deleteBtn.textContent = 'Deleting...';
      try {
        await window.artifex.deleteModel(model.path);
        card.classList.add('model-card-removing');
        setTimeout(() => {
          card.remove();
          // Check if grid is now empty
          if (modelsGrid && modelsGrid.querySelectorAll('.model-card').length === 0) {
            modelsGrid.innerHTML = `
              <div class="models-empty">
                No models found in the models/ directory.
              </div>
            `;
          }
        }, 300);
      } catch (err) {
        deleteBtn.disabled = false;
        deleteBtn.textContent = 'Delete';
        await showModal('Delete Failed', `Could not delete model: ${err.message || err}`);
      }
    }
  };
  actions.appendChild(deleteBtn);

  return card;
}

function createOllamaModelCard(model) {
  const card = document.createElement('div');
  card.className = 'model-card';
  card.dataset.source = 'ollama';
  card.dataset.name = model.name;

  const quantBadge = model.quantLevel || 'GGUF';

  card.innerHTML = `
    <div class="model-card-header">
      <div class="model-name">${escapeHtml(model.name)}</div>
      <span class="model-badge badge-quant">${escapeHtml(quantBadge)}</span>
    </div>
    <div class="model-source-row">
      <span class="model-source-badge source-ollama">Ollama</span>
    </div>
    <div class="model-size">${escapeHtml(model.sizeFormatted)}</div>
    <div class="model-meta-grid">
      <div class="model-meta-item">
        <span class="model-meta-label">Family</span>
        <span class="model-meta-value">${escapeHtml(model.family)}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Params</span>
        <span class="model-meta-value">${escapeHtml(model.paramSize || '—')}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Format</span>
        <span class="model-meta-value">${escapeHtml((model.format || 'gguf').toUpperCase())}</span>
      </div>
    </div>
    <div class="model-card-actions"></div>
  `;

  const actions = card.querySelector('.model-card-actions');
  const deleteBtn = document.createElement('button');
  deleteBtn.className = 'btn btn-danger btn-sm';
  deleteBtn.textContent = 'Delete';
  deleteBtn.onclick = async (e) => {
    e.stopPropagation();
    const confirmed = await showModal(
      'Delete Ollama Model',
      `Permanently delete "${model.name}" from Ollama? This will free ${model.sizeFormatted}.`
    );
    if (!confirmed) return;
    deleteBtn.disabled = true;
    deleteBtn.textContent = 'Deleting...';
    try {
      await window.artifex.deleteOllamaModel(model.name);
      card.classList.add('model-card-removing');
      setTimeout(() => {
        card.remove();
        if (modelsGrid && modelsGrid.querySelectorAll('.model-card').length === 0) {
          modelsGrid.innerHTML = `
            <div class="models-empty">
              No models found.
            </div>
          `;
        }
      }, 300);
    } catch (err) {
      deleteBtn.disabled = false;
      deleteBtn.textContent = 'Delete';
      await showModal('Delete Failed', `Could not delete model: ${err.message || err}`);
    }
  };
  actions.appendChild(deleteBtn);

  return card;
}

function createLlamaCppModelCard(model) {
  const card = document.createElement('div');
  card.className = 'model-card';
  card.dataset.source = 'llama_cpp';
  card.dataset.name = model.name;

  // Visual cue when the declared GGUF doesn't actually exist on disk
  if (model.ggufMissing) {
    card.classList.add('model-card-warning');
  }

  const sizeLabel = model.ggufMissing ? 'GGUF missing' : escapeHtml(model.sizeFormatted);
  const ctxLabel = model.numCtx == null ? 'auto' : model.numCtx.toLocaleString();
  const flagBadges = (model.flagsSummary || [])
    .map((f) => `<span class="flag-badge">${escapeHtml(f)}</span>`)
    .join('');

  card.innerHTML = `
    <div class="model-card-header">
      <div class="model-name">${escapeHtml(model.name)}</div>
      <span class="model-badge ${model.hasMmproj ? 'badge-vision' : 'badge-quant'}">${model.hasMmproj ? 'VL' : 'GGUF'}</span>
    </div>
    <div class="model-source-row">
      <span class="model-source-badge source-llama-cpp">llama.cpp</span>
      ${model.hasMmproj ? '<span class="model-source-badge source-mmproj">+ mmproj</span>' : ''}
    </div>
    <div class="model-size">${sizeLabel}</div>
    <div class="model-meta-grid model-meta-grid-2col">
      <div class="model-meta-item">
        <span class="model-meta-label">Port</span>
        <span class="model-meta-value">${model.port}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Context</span>
        <span class="model-meta-value">${ctxLabel}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">GPU layers</span>
        <span class="model-meta-value">${model.numGpuLayers}</span>
      </div>
      <div class="model-meta-item">
        <span class="model-meta-label">Vision</span>
        <span class="model-meta-value">${model.hasMmproj ? 'yes' : 'no'}</span>
      </div>
    </div>
    ${flagBadges ? `<div class="flag-row">${flagBadges}</div>` : ''}
    ${model.ggufMissing ? `
      <div class="card-warning-row">
        GGUF path declared in <code>llama_cpp_config.json</code> doesn't exist on disk.
        Either download it or fix the <code>path</code> field.
      </div>
    ` : ''}
  `;

  return card;
}

function buildCacheControl(model, card) {
  const wrapper = document.createElement('label');
  wrapper.className = 'cache-toggle';

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = !!model.noCache;

  const labelText = document.createElement('span');
  labelText.className = 'cache-toggle-label';

  const status = document.createElement('span');
  status.className = 'cache-toggle-status';

  function render(state) {
    labelText.textContent = 'Skip NF4 cache';
    if (state.noCache) {
      status.textContent = 'on disk: 0 B (cache disabled)';
      status.classList.add('cache-disabled');
    } else if (state.cacheSizeBytes > 0) {
      status.textContent = `cached: ${state.cacheSizeFormatted}`;
      status.classList.remove('cache-disabled');
    } else {
      status.textContent = 'no cache yet';
      status.classList.remove('cache-disabled');
    }
  }
  render({
    noCache: model.noCache,
    cacheSizeBytes: model.cacheSizeBytes,
    cacheSizeFormatted: model.cacheSizeFormatted,
  });

  checkbox.onchange = async () => {
    const desired = checkbox.checked;
    checkbox.disabled = true;
    const prevText = status.textContent;
    status.textContent = desired ? 'disabling cache…' : 'enabling cache…';
    try {
      const result = await window.artifex.setNoCache(model.path, desired);
      // Mutate in-place so subsequent renders see fresh state without a
      // full grid refresh.
      model.noCache = result.noCache;
      model.cachePath = result.cachePath;
      model.cacheSizeBytes = result.cacheSizeBytes;
      model.cacheSizeFormatted = result.cacheSizeFormatted;
      render(result);
    } catch (err) {
      checkbox.checked = !desired;
      status.textContent = prevText;
      await showModal(
        'Cache Toggle Failed',
        `Could not ${desired ? 'disable' : 'enable'} the NF4 cache: ${err?.message || err}`,
      );
    } finally {
      checkbox.disabled = false;
    }
  };

  wrapper.appendChild(checkbox);
  wrapper.appendChild(labelText);
  wrapper.appendChild(status);
  return wrapper;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function injectModelsPanelStyles() {
  if (document.getElementById('models-panel-styles')) return;

  const style = document.createElement('style');
  style.id = 'models-panel-styles';
  style.textContent = `
    .models-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
      gap: 16px;
      padding: 16px;
    }

    .model-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .model-card:hover {
      border-color: var(--accent);
      transform: translateY(-1px);
    }

    .model-card-removing {
      opacity: 0;
      transform: scale(0.95);
      transition: opacity 0.3s ease, transform 0.3s ease;
    }

    .model-card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 8px;
    }

    .model-name {
      font-weight: 600;
      font-size: 15px;
      color: var(--text);
      word-break: break-word;
      line-height: 1.3;
    }

    .model-badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 10px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      white-space: nowrap;
      flex-shrink: 0;
    }
    .model-source-row {
      display: flex;
      gap: 6px;
      margin-top: -4px;
    }
    .model-source-badge {
      display: inline-block;
      padding: 1px 8px;
      border-radius: 8px;
      font-size: 9px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }
    .source-transformers {
      background: rgba(0, 240, 255, 0.08);
      color: var(--accent);
      border: 1px solid rgba(0, 240, 255, 0.25);
    }
    .source-ollama {
      background: rgba(80, 200, 120, 0.10);
      color: #50c878;
      border: 1px solid rgba(80, 200, 120, 0.30);
    }
    .source-llama-cpp {
      background: rgba(255, 140, 50, 0.10);
      color: #ff9533;
      border: 1px solid rgba(255, 140, 50, 0.30);
    }
    .source-mmproj {
      background: rgba(176, 96, 255, 0.10);
      color: var(--accent2);
      border: 1px solid rgba(176, 96, 255, 0.30);
    }
    .badge-vision {
      background: rgba(176, 96, 255, 0.15);
      color: var(--accent2);
    }

    .model-card-warning {
      border-color: var(--warn);
    }
    .card-warning-row {
      font-size: 11px;
      color: var(--warn);
      background: rgba(255, 204, 0, 0.06);
      border: 1px solid rgba(255, 204, 0, 0.25);
      border-radius: 6px;
      padding: 6px 8px;
      line-height: 1.4;
    }
    .card-warning-row code {
      font-family: var(--mono);
      font-size: 10px;
      background: rgba(255, 204, 0, 0.10);
      padding: 1px 4px;
      border-radius: 3px;
    }

    .model-meta-grid-2col {
      grid-template-columns: repeat(2, 1fr);
    }

    .flag-row {
      display: flex;
      flex-wrap: wrap;
      gap: 4px;
      margin-top: 2px;
    }
    .flag-badge {
      display: inline-block;
      padding: 1px 6px;
      border-radius: 4px;
      font-family: var(--mono);
      font-size: 10px;
      color: var(--dim);
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid var(--border);
    }
    .badge-bf16 {
      background: rgba(0, 240, 255, 0.12);
      color: var(--accent);
    }
    .badge-quant {
      background: rgba(176, 96, 255, 0.15);
      color: var(--accent2);
    }
    .badge-mixed {
      background: rgba(255, 204, 0, 0.12);
      color: var(--warn);
    }

    .model-size {
      font-family: var(--mono);
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
      letter-spacing: -0.5px;
    }

    .model-meta-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px 12px;
    }

    .model-meta-item {
      display: flex;
      flex-direction: column;
      gap: 1px;
    }

    .model-meta-label {
      font-size: 10px;
      color: var(--dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .model-meta-value {
      font-family: var(--mono);
      font-size: 13px;
      color: var(--text);
    }

    .model-cache-row {
      display: flex;
      align-items: center;
      margin-top: 2px;
    }
    .cache-toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--text);
      cursor: pointer;
      user-select: none;
    }
    .cache-toggle input[type="checkbox"] {
      cursor: pointer;
      margin: 0;
    }
    .cache-toggle input[type="checkbox"]:disabled {
      cursor: wait;
      opacity: 0.6;
    }
    .cache-toggle-label {
      font-weight: 500;
    }
    .cache-toggle-status {
      color: var(--dim);
      font-family: var(--mono);
      font-size: 11px;
    }
    .cache-toggle-status.cache-disabled {
      color: var(--accent2);
    }

    .model-card-actions {
      display: flex;
      gap: 6px;
      margin-top: 4px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }

    .models-loading,
    .models-empty {
      grid-column: 1 / -1;
      text-align: center;
      padding: 48px 16px;
      color: var(--dim);
      font-size: 14px;
    }
  `;
  document.head.appendChild(style);
}
