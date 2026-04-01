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

async function loadModels() {
  if (modelsLoading || !modelsGrid) return;
  modelsLoading = true;

  modelsGrid.innerHTML = `
    <div class="models-loading">Scanning models directory...</div>
  `;

  try {
    const models = await window.artifex.scanModels();

    modelsGrid.innerHTML = '';

    if (!models || models.length === 0) {
      modelsGrid.innerHTML = `
        <div class="models-empty">
          No models found in the models/ directory.
        </div>
      `;
      modelsLoading = false;
      return;
    }

    for (const model of models) {
      modelsGrid.appendChild(createModelCard(model));
    }
  } catch (err) {
    modelsGrid.innerHTML = `
      <div class="models-empty" style="color:var(--error)">
        Failed to scan models: ${err.message || err}
      </div>
    `;
  }

  modelsLoading = false;
}

function createModelCard(model) {
  const card = document.createElement('div');
  card.className = 'model-card';
  card.dataset.path = model.path;

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
    <div class="model-card-actions"></div>
  `;

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
