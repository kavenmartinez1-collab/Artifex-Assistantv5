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
    <div class="service-config" style="display:none"></div>
    <div class="actions"></div>
  `;

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
      if (!serviceConfigs[e.target.dataset.serviceId]) serviceConfigs[e.target.dataset.serviceId] = {};
      serviceConfigs[e.target.dataset.serviceId][e.target.dataset.optKey] = e.target.value;
    });

    wrapper.appendChild(input);
    row.appendChild(wrapper);
  }

  configDiv.appendChild(row);
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
