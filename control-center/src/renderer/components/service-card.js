// Service card component — creates DOM element for one service
// Includes configurable options (port, backend, model) shown when stopped

const STATUS_CLASSES = {
  running: 'running',
  stopped: 'stopped',
  starting: 'starting',
  stopping: 'starting',
  error: 'error',
};

// Per-service option definitions (loaded from backend)
let serviceOptionDefs = {};

// Per-service current config values (persisted during session)
const serviceConfigs = {};

// Cache of model names by backend, populated lazily by model-select fields.
// Shape: { transformers: ['name', ...], ollama: ['name', ...] }
let modelNamesByBackend = null;
let modelNamesPromise = null;

/** Load option definitions from the backend once */
async function loadServiceOptions() {
  if (Object.keys(serviceOptionDefs).length > 0) return;
  try {
    const defs = await window.artifex.getServiceOptions();
    for (const d of defs) {
      if (d.options && d.options.length > 0) {
        serviceOptionDefs[d.id] = d.options;
      }
    }
  } catch { /* backend may not support this yet */ }
}

/** Lazily fetch and cache the names of models available on each backend. */
function getModelNamesByBackend() {
  if (modelNamesByBackend) return Promise.resolve(modelNamesByBackend);
  if (modelNamesPromise) return modelNamesPromise;
  modelNamesPromise = (async () => {
    try {
      modelNamesByBackend = await window.artifex.listModelNamesByBackend();
    } catch {
      modelNamesByBackend = { transformers: [], ollama: [] };
    }
    modelNamesPromise = null;
    return modelNamesByBackend;
  })();
  return modelNamesPromise;
}

/** Force a refresh of the cached model names — call after install/delete. */
function invalidateModelNamesCache() {
  modelNamesByBackend = null;
  modelNamesPromise = null;
}

function createServiceCard(service) {
  const card = document.createElement('div');
  card.className = 'service-card';
  card.dataset.id = service.id;

  card.innerHTML = `
    <div class="name">${service.name}</div>
    <div class="port">:${service.port}</div>
    <div>
      <span class="status-badge ${STATUS_CLASSES[service.status] || 'stopped'}">${service.status}</span>
    </div>
    <div class="meta pid-info"></div>
    <div class="meta uptime-info"></div>
    <div class="service-warning" style="display:none"></div>
    <div class="service-config" style="display:none"></div>
    <div class="actions"></div>
  `;

  // Web gateway: show Docker conflict warning
  if (service.id === 'web-gateway') {
    checkWebGatewayDockerConflict(card);
  }

  updateCardActions(card, service);
  updateCardMeta(card, service);
  updateCardConfig(card, service);
  return card;
}

function updateServiceCard(card, service) {
  const badge = card.querySelector('.status-badge');
  badge.className = `status-badge ${STATUS_CLASSES[service.status] || 'stopped'}`;
  badge.textContent = service.status;

  // Update port display if overridden
  const portEl = card.querySelector('.port');
  const config = serviceConfigs[service.id];
  const effectivePort = config?.['--port'] || service.port;
  portEl.textContent = `:${effectivePort}`;

  updateCardActions(card, service);
  updateCardMeta(card, service);
  updateCardConfig(card, service);
}

/** Show config inputs when stopped, hide when running */
function updateCardConfig(card, service) {
  const configDiv = card.querySelector('.service-config');
  if (!configDiv) return;

  const opts = serviceOptionDefs[service.id];
  if (!opts || opts.length === 0) {
    configDiv.style.display = 'none';
    return;
  }

  // Only show config when stopped or error
  if (service.status === 'running' || service.status === 'starting' || service.status === 'stopping') {
    configDiv.style.display = 'none';
    return;
  }

  configDiv.style.display = 'block';

  // Only rebuild if empty (preserve user edits)
  if (configDiv.children.length > 0) return;

  // Initialize config values with defaults
  if (!serviceConfigs[service.id]) {
    serviceConfigs[service.id] = {};
    for (const opt of opts) {
      serviceConfigs[service.id][opt.key] = opt.default;
    }
  }

  configDiv.innerHTML = '';
  const row = document.createElement('div');
  row.style.cssText = 'display:flex;flex-wrap:wrap;gap:8px;padding:6px 0;border-top:1px solid var(--border);margin-top:4px;';

  // Track inputs by option key so dependent fields can find each other
  const inputsByKey = {};

  for (const opt of opts) {
    const wrapper = document.createElement('div');
    wrapper.style.cssText = 'display:flex;flex-direction:column;gap:2px;';

    const label = document.createElement('label');
    label.textContent = opt.label;
    label.style.cssText = 'font-size:10px;color:var(--dim);text-transform:uppercase;';
    wrapper.appendChild(label);

    let input;
    if (opt.type === 'select' && opt.choices) {
      input = document.createElement('select');
      input.style.cssText = 'background:var(--bg-secondary);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:11px;';
      for (const choice of opt.choices) {
        const optEl = document.createElement('option');
        optEl.value = choice;
        optEl.textContent = choice;
        if (choice === serviceConfigs[service.id][opt.key]) optEl.selected = true;
        input.appendChild(optEl);
      }
    } else if (opt.type === 'model-select') {
      // Dropdown that filters by the current value of `dependsOn` (e.g. backend).
      // Populated asynchronously after the cache is loaded.
      input = document.createElement('select');
      input.style.cssText = 'background:var(--bg-secondary);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:11px;min-width:160px;';
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = 'Loading models...';
      input.appendChild(placeholder);
      input.dataset.dependsOn = opt.dependsOn || '';
      input.dataset.modelSelect = '1';
    } else {
      input = document.createElement('input');
      input.type = opt.type === 'number' ? 'number' : 'text';
      input.value = serviceConfigs[service.id][opt.key] || '';
      input.placeholder = opt.default;
      input.style.cssText = 'background:var(--bg-secondary);color:var(--text);border:1px solid var(--border);border-radius:3px;padding:2px 4px;font-size:11px;width:' + (opt.type === 'number' ? '60px' : '120px') + ';';
    }

    input.dataset.optKey = opt.key;
    input.dataset.serviceId = service.id;
    input.addEventListener('change', (e) => {
      const sid = e.target.dataset.serviceId;
      const key = e.target.dataset.optKey;
      if (!serviceConfigs[sid]) serviceConfigs[sid] = {};
      serviceConfigs[sid][key] = e.target.value;

      // If this field is the dependsOn target of any model-select, refresh it
      for (const dep of Object.values(inputsByKey)) {
        if (dep.dataset.modelSelect && dep.dataset.dependsOn === key) {
          populateModelSelect(dep, e.target.value, service.id);
        }
      }
    });

    wrapper.appendChild(input);
    row.appendChild(wrapper);
    inputsByKey[opt.key] = input;
  }

  configDiv.appendChild(row);

  // Populate any model-select dropdowns now that all sibling inputs exist
  for (const opt of opts) {
    if (opt.type !== 'model-select') continue;
    const select = inputsByKey[opt.key];
    const dependsKey = opt.dependsOn;
    const backendValue = dependsKey
      ? (serviceConfigs[service.id][dependsKey] || inputsByKey[dependsKey]?.value || '')
      : '';
    populateModelSelect(select, backendValue, service.id);
  }
}

/**
 * Fill a model-select dropdown with the names available on the given backend.
 * Uses the cached model name map; falls back to a free-text input if the
 * cache is unreachable, so a broken Ollama daemon doesn't strand the user.
 */
async function populateModelSelect(selectEl, backendValue, serviceId) {
  const cache = await getModelNamesByBackend();
  const names = (cache && cache[backendValue]) || [];

  const previous = serviceConfigs[serviceId]?.[selectEl.dataset.optKey] || '';
  selectEl.innerHTML = '';

  if (names.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = backendValue
      ? `(no ${backendValue} models found)`
      : '(select a backend first)';
    selectEl.appendChild(opt);
    // Persist empty so we don't pass a stale model name to the backend
    if (serviceConfigs[serviceId]) {
      serviceConfigs[serviceId][selectEl.dataset.optKey] = '';
    }
    return;
  }

  // Default sentinel — lets the API server pick the active model
  const defaultOpt = document.createElement('option');
  defaultOpt.value = '';
  defaultOpt.textContent = '(default)';
  selectEl.appendChild(defaultOpt);

  for (const name of names) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    if (name === previous) opt.selected = true;
    selectEl.appendChild(opt);
  }

  if (!serviceConfigs[serviceId]) serviceConfigs[serviceId] = {};
  // If the previously-selected model isn't in this backend's list, clear it
  if (previous && !names.includes(previous)) {
    serviceConfigs[serviceId][selectEl.dataset.optKey] = '';
  }
}

function updateCardActions(card, service) {
  const actions = card.querySelector('.actions');
  actions.innerHTML = '';

  if (service.status === 'running') {
    const stopBtn = document.createElement('button');
    stopBtn.className = 'btn btn-danger btn-sm';
    stopBtn.textContent = 'Stop';
    stopBtn.onclick = () => window.artifex.stopService(service.id);
    actions.appendChild(stopBtn);

    const restartBtn = document.createElement('button');
    restartBtn.className = 'btn btn-sm';
    restartBtn.textContent = 'Restart';
    restartBtn.onclick = () => window.artifex.restartService(service.id);
    actions.appendChild(restartBtn);
  } else if (service.status === 'stopped' || service.status === 'error') {
    const startBtn = document.createElement('button');
    startBtn.className = 'btn btn-primary btn-sm';
    startBtn.textContent = 'Start';
    startBtn.onclick = () => {
      // Pass config overrides to start
      const overrides = serviceConfigs[service.id] || {};
      window.artifex.startService(service.id, overrides);
    };
    actions.appendChild(startBtn);
  }

  if (service.status === 'error' && service.errorMessage) {
    const errEl = document.createElement('div');
    errEl.className = 'error-msg';
    errEl.style.cssText = 'font-size:10px;color:var(--error);margin-top:4px;word-break:break-all;';
    errEl.textContent = service.errorMessage;
    actions.appendChild(errEl);
  }
}

/** Check if Docker is running a web-gateway container and show warning */
async function checkWebGatewayDockerConflict(card) {
  const warningEl = card.querySelector('.service-warning');
  if (!warningEl || !window.artifex?.docker) return;
  try {
    const isDocker = await window.artifex.docker.isDockerInstalled();
    if (!isDocker) return;
    const containers = await window.artifex.docker.listContainers();
    const gwContainer = containers.find(c =>
      c.name?.includes('web-gateway') || c.image?.includes('web-gateway')
    );
    if (gwContainer && gwContainer.status === 'running') {
      warningEl.style.display = 'block';
      warningEl.style.cssText = 'display:block;font-size:10px;color:#ff9800;background:#2a2200;border:1px solid #ff9800;border-radius:3px;padding:4px 6px;margin:4px 0;';
      warningEl.innerHTML = '⚠ Docker web-gateway is running on port 8080. <b>Do not start this local gateway</b> — it will conflict. Use Docker\'s gateway instead.';
    }
  } catch { /* Docker not available — no conflict */ }
}

function updateCardMeta(card, service) {
  const pidEl = card.querySelector('.pid-info');
  const uptimeEl = card.querySelector('.uptime-info');

  if (service.status === 'running' && service.pid) {
    pidEl.textContent = `PID: ${service.pid}`;
    if (service.startTime) {
      const elapsed = Math.floor((Date.now() - service.startTime) / 1000);
      const mins = Math.floor(elapsed / 60);
      const hrs = Math.floor(mins / 60);
      uptimeEl.textContent = hrs > 0
        ? `Up: ${hrs}h ${mins % 60}m`
        : `Up: ${mins}m ${elapsed % 60}s`;
    }
  } else {
    pidEl.textContent = '';
    uptimeEl.textContent = '';
  }
}
