// Docker panel — container overview, compose controls, resource monitoring

let dockerRefreshTimer = null;
let dockerContainerCards = {};
let dockerIsInstalled = false;
let dockerComposePathInput = null;
let dockerLogsOverlay = null;
let dockerRefreshInProgress = false;

async function initDockerPanel(container) {
  // Clean up previous timer if re-initializing
  if (dockerRefreshTimer) {
    clearInterval(dockerRefreshTimer);
    dockerRefreshTimer = null;
  }
  dockerContainerCards = {};

  container.innerHTML = '';

  // ── Header bar ──
  const header = document.createElement('div');
  header.className = 'panel-header';
  header.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px">
      <h2>Docker</h2>
      <span id="docker-status-dot" style="
        width:10px; height:10px; border-radius:50%;
        background:#f44336; display:inline-block;
      "></span>
      <span id="docker-status-text" style="font-size:12px;color:#888">Checking...</span>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <button class="btn btn-primary btn-sm" id="docker-compose-up-btn" disabled>Compose Up</button>
      <button class="btn btn-danger btn-sm" id="docker-compose-down-btn" disabled>Compose Down</button>
      <button class="btn btn-sm" id="docker-refresh-btn">Refresh</button>
    </div>
  `;
  container.appendChild(header);

  // ── Not-installed message (hidden by default) ──
  const notInstalledMsg = document.createElement('div');
  notInstalledMsg.id = 'docker-not-installed';
  notInstalledMsg.style.cssText = 'display:none;padding:48px 32px;text-align:center;';
  notInstalledMsg.innerHTML = `
    <div style="font-size:48px;margin-bottom:16px;color:#2a2d3e">&#9899;</div>
    <div style="font-size:18px;color:#e0e0e0;margin-bottom:8px">Docker is not installed</div>
    <div style="font-size:13px;color:#888;max-width:420px;margin:0 auto;line-height:1.6">
      Docker Desktop or Docker Engine was not detected on this system.
      Install Docker from <span style="color:#00f0ff">docker.com/get-started</span>
      and make sure the Docker daemon is running, then click Refresh.
    </div>
    <button class="btn btn-primary btn-sm" id="docker-retry-btn" style="margin-top:20px">
      Retry Detection
    </button>
  `;
  container.appendChild(notInstalledMsg);

  // ── Container card grid ──
  const grid = document.createElement('div');
  grid.className = 'card-grid';
  grid.id = 'docker-grid';
  container.appendChild(grid);

  // ── Compose file path bar ──
  const composeBar = document.createElement('div');
  composeBar.id = 'docker-compose-bar';
  composeBar.style.cssText = `
    padding: 12px 16px;
    border-top: 1px solid #2a2d3e;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
  `;
  composeBar.innerHTML = `
    <span style="color:#888;white-space:nowrap">Compose file:</span>
    <input type="text" id="docker-compose-path"
      placeholder="path/to/docker-compose.yml"
      style="
        flex:1; background:#0f111a; border:1px solid #2a2d3e; color:#e0e0e0;
        padding:4px 8px; border-radius:4px; font-size:12px;
        font-family:Consolas,'Cascadia Code','Fira Code',monospace;
      "
    />
  `;
  container.appendChild(composeBar);

  // ── Logs overlay (hidden, shown on demand) ──
  dockerLogsOverlay = document.createElement('div');
  dockerLogsOverlay.id = 'docker-logs-overlay';
  dockerLogsOverlay.style.cssText = `
    display:none; position:absolute; top:0; left:0; right:0; bottom:0;
    background:rgba(15,17,26,0.95); z-index:100;
    flex-direction:column; padding:16px;
  `;
  dockerLogsOverlay.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <span id="docker-logs-title" style="font-size:14px;font-weight:600;color:#00f0ff">Container Logs</span>
      <button class="btn btn-sm" id="docker-logs-close-btn">Close</button>
    </div>
    <pre id="docker-logs-content" style="
      flex:1; overflow:auto; background:#0a0c14; border:1px solid #2a2d3e;
      border-radius:4px; padding:8px; font-size:12px; color:#888;
      font-family:Consolas,'Cascadia Code','Fira Code',monospace;
      line-height:1.4; white-space:pre-wrap; margin:0;
    ">Loading...</pre>
  `;
  container.appendChild(dockerLogsOverlay);

  // ── Style injection for docker-specific styles ──
  injectDockerStyles();

  // ── Wire up controls ──
  dockerComposePathInput = document.getElementById('docker-compose-path');

  document.getElementById('docker-compose-up-btn').onclick = async () => {
    const p = dockerComposePathInput.value.trim();
    if (!p) return;
    try {
      document.getElementById('docker-compose-up-btn').disabled = true;
      document.getElementById('docker-compose-up-btn').textContent = 'Starting...';
      await window.artifex.docker.composeUp(p);
      await refreshDockerPanel();
    } catch (err) {
      showDockerToast('Compose Up failed: ' + err.message, true);
    } finally {
      document.getElementById('docker-compose-up-btn').disabled = false;
      document.getElementById('docker-compose-up-btn').textContent = 'Compose Up';
    }
  };

  document.getElementById('docker-compose-down-btn').onclick = async () => {
    const p = dockerComposePathInput.value.trim();
    if (!p) return;
    try {
      document.getElementById('docker-compose-down-btn').disabled = true;
      document.getElementById('docker-compose-down-btn').textContent = 'Stopping...';
      await window.artifex.docker.composeDown(p);
      await refreshDockerPanel();
    } catch (err) {
      showDockerToast('Compose Down failed: ' + err.message, true);
    } finally {
      document.getElementById('docker-compose-down-btn').disabled = false;
      document.getElementById('docker-compose-down-btn').textContent = 'Compose Down';
    }
  };

  document.getElementById('docker-refresh-btn').onclick = () => refreshDockerPanel();
  document.getElementById('docker-retry-btn').onclick = () => refreshDockerPanel();
  document.getElementById('docker-logs-close-btn').onclick = () => {
    dockerLogsOverlay.style.display = 'none';
  };

  // ── Initial load ──
  await refreshDockerPanel();

  // ── Auto-refresh every 10 seconds while panel is visible ──
  dockerRefreshTimer = setInterval(() => {
    if (document.getElementById('docker-grid')) {
      refreshDockerPanel();
    } else {
      clearInterval(dockerRefreshTimer);
      dockerRefreshTimer = null;
    }
  }, 10000);
}

/** Refresh the entire Docker panel state */
async function refreshDockerPanel() {
  if (dockerRefreshInProgress) return;
  dockerRefreshInProgress = true;
  try { await _refreshDockerPanelInner(); } finally { dockerRefreshInProgress = false; }
}

async function _refreshDockerPanelInner() {
  const statusDot = document.getElementById('docker-status-dot');
  const statusText = document.getElementById('docker-status-text');
  const notInstalled = document.getElementById('docker-not-installed');
  const grid = document.getElementById('docker-grid');
  const composeUpBtn = document.getElementById('docker-compose-up-btn');
  const composeDownBtn = document.getElementById('docker-compose-down-btn');

  if (!statusDot || !grid) return;

  try {
    dockerIsInstalled = await window.artifex.docker.isDockerInstalled();
  } catch {
    dockerIsInstalled = false;
  }

  if (!dockerIsInstalled) {
    statusDot.style.background = '#f44336';
    statusText.textContent = 'Not detected';
    statusText.style.color = '#f44336';
    notInstalled.style.display = 'block';
    grid.style.display = 'none';
    composeUpBtn.disabled = true;
    composeDownBtn.disabled = true;
    return;
  }

  // Docker is available
  statusDot.style.background = '#4caf50';
  statusText.textContent = 'Connected';
  statusText.style.color = '#4caf50';
  notInstalled.style.display = 'none';
  grid.style.display = 'flex';
  composeUpBtn.disabled = false;
  composeDownBtn.disabled = false;

  // Fetch containers
  let containers = [];
  try {
    containers = await window.artifex.docker.listContainers();
  } catch (err) {
    showDockerToast('Failed to list containers: ' + err.message, true);
    return;
  }

  // Reconcile cards
  const seenIds = new Set();
  for (const c of containers) {
    seenIds.add(c.id);
    if (dockerContainerCards[c.id]) {
      updateDockerCard(dockerContainerCards[c.id], c);
    } else {
      const card = createDockerCard(c);
      dockerContainerCards[c.id] = card;
      grid.appendChild(card);
    }
  }

  // Remove stale cards
  for (const id of Object.keys(dockerContainerCards)) {
    if (!seenIds.has(id)) {
      dockerContainerCards[id].remove();
      delete dockerContainerCards[id];
    }
  }

  // Show empty state (clear stale card references)
  if (containers.length === 0 && grid.children.length === 0) {
    dockerContainerCards = {};
    grid.innerHTML = `
      <div style="padding:32px;text-align:center;color:#888;width:100%">
        No containers found. Start a container or use Compose Up.
      </div>
    `;
  }
}

/** Create a container card DOM element */
function createDockerCard(info) {
  const card = document.createElement('div');
  card.className = 'docker-card';
  card.dataset.id = info.id;

  card.innerHTML = buildDockerCardInner(info);
  wireDockerCardButtons(card, info);
  return card;
}

/** Update an existing container card */
function updateDockerCard(card, info) {
  card.innerHTML = buildDockerCardInner(info);
  wireDockerCardButtons(card, info);
}

/** Build the inner HTML for a container card */
function buildDockerCardInner(info) {
  const statusColor = info.status === 'running' ? '#4caf50'
    : info.status === 'restarting' ? '#ff9800'
    : info.status === 'created' ? '#888'
    : '#f44336';

  const statusBgColor = info.status === 'running' ? 'rgba(76,175,80,0.15)'
    : info.status === 'restarting' ? 'rgba(255,152,0,0.15)'
    : info.status === 'created' ? 'rgba(136,136,136,0.15)'
    : 'rgba(244,67,54,0.15)';

  const portsHtml = info.ports.length > 0
    ? info.ports.map(function(p) {
        return '<span class="docker-port">' + escapeDockerHtml(p) + '</span>';
      }).join(' ')
    : '<span style="color:#888;font-size:11px">No ports</span>';

  return '' +
    '<div class="docker-card-header">' +
      '<span class="docker-card-name">' + escapeDockerHtml(info.name) + '</span>' +
      '<span class="docker-status-badge" style="background:' + statusBgColor + ';color:' + statusColor + '">' +
        escapeDockerHtml(info.status) +
      '</span>' +
    '</div>' +
    '<div class="docker-card-image">' + escapeDockerHtml(info.image) + '</div>' +
    '<div class="docker-card-ports">' + portsHtml + '</div>' +
    '<div class="docker-card-stats">' +
      '<div class="docker-stat">' +
        '<span class="docker-stat-label">CPU</span>' +
        '<span class="docker-stat-value">' + escapeDockerHtml(info.cpu) + '</span>' +
      '</div>' +
      '<div class="docker-stat">' +
        '<span class="docker-stat-label">Memory</span>' +
        '<span class="docker-stat-value">' + escapeDockerHtml(info.memory) + '</span>' +
      '</div>' +
    '</div>' +
    '<div class="docker-card-uptime">' + escapeDockerHtml(info.uptime) + '</div>' +
    '<div class="docker-card-actions"></div>';
}

/** Wire up Start/Stop/Logs buttons on a card */
function wireDockerCardButtons(card, info) {
  var actions = card.querySelector('.docker-card-actions');
  if (!actions) return;

  if (info.status === 'running' || info.status === 'restarting') {
    var stopBtn = document.createElement('button');
    stopBtn.className = 'btn btn-danger btn-sm';
    stopBtn.textContent = 'Stop';
    stopBtn.onclick = async function() {
      stopBtn.disabled = true;
      stopBtn.textContent = 'Stopping...';
      try {
        await window.artifex.docker.stopContainer(info.name || info.id);
        await refreshDockerPanel();
      } catch (err) {
        showDockerToast('Stop failed: ' + err.message, true);
        stopBtn.disabled = false;
        stopBtn.textContent = 'Stop';
      }
    };
    actions.appendChild(stopBtn);
  } else {
    var startBtn = document.createElement('button');
    startBtn.className = 'btn btn-primary btn-sm';
    startBtn.textContent = 'Start';
    startBtn.onclick = async function() {
      startBtn.disabled = true;
      startBtn.textContent = 'Starting...';
      try {
        await window.artifex.docker.startContainer(info.name || info.id);
        await refreshDockerPanel();
      } catch (err) {
        showDockerToast('Start failed: ' + err.message, true);
        startBtn.disabled = false;
        startBtn.textContent = 'Start';
      }
    };
    actions.appendChild(startBtn);
  }

  var logsBtn = document.createElement('button');
  logsBtn.className = 'btn btn-sm';
  logsBtn.textContent = 'Logs';
  logsBtn.onclick = async function() {
    await showContainerLogs(info.name || info.id);
  };
  actions.appendChild(logsBtn);
}

/** Show the logs overlay for a container */
async function showContainerLogs(nameOrId) {
  if (!dockerLogsOverlay) return;

  var title = document.getElementById('docker-logs-title');
  var content = document.getElementById('docker-logs-content');

  title.textContent = 'Logs: ' + nameOrId;
  content.textContent = 'Loading...';
  dockerLogsOverlay.style.display = 'flex';

  try {
    var logs = await window.artifex.docker.getContainerLogs(nameOrId, 500);
    content.textContent = logs || '(no output)';
    content.scrollTop = content.scrollHeight;
  } catch (err) {
    content.textContent = 'Error fetching logs: ' + err.message;
  }
}

/** Simple toast notification for the Docker panel */
function showDockerToast(message, isError) {
  var existing = document.getElementById('docker-toast');
  if (existing) existing.remove();

  var toast = document.createElement('div');
  toast.id = 'docker-toast';
  toast.style.cssText = '' +
    'position:fixed;bottom:48px;right:16px;padding:10px 16px;' +
    'border-radius:6px;font-size:12px;z-index:200;max-width:400px;' +
    'background:' + (isError ? '#f44336' : '#4caf50') + ';' +
    'color:#fff;box-shadow:0 4px 12px rgba(0,0,0,0.4);';
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(function() {
    if (toast.parentNode) toast.remove();
  }, 4000);
}

/** Escape HTML for safe rendering */
function escapeDockerHtml(text) {
  var div = document.createElement('div');
  div.textContent = text || '';
  return div.innerHTML;
}

/** Inject docker-specific CSS once */
function injectDockerStyles() {
  if (document.getElementById('docker-panel-styles')) return;

  var style = document.createElement('style');
  style.id = 'docker-panel-styles';
  style.textContent = '' +
    /* Card */
    '.docker-card {' +
    '  background: #1a1d2e;' +
    '  border: 1px solid #2a2d3e;' +
    '  border-radius: 8px;' +
    '  padding: 16px;' +
    '  display: flex;' +
    '  flex-direction: column;' +
    '  gap: 8px;' +
    '  min-width: 260px;' +
    '  max-width: 340px;' +
    '  flex: 1 1 280px;' +
    '  transition: border-color 0.15s;' +
    '}' +
    '.docker-card:hover {' +
    '  border-color: #00f0ff44;' +
    '}' +

    /* Card header */
    '.docker-card-header {' +
    '  display: flex;' +
    '  justify-content: space-between;' +
    '  align-items: center;' +
    '  gap: 8px;' +
    '}' +
    '.docker-card-name {' +
    '  font-weight: 600;' +
    '  font-size: 14px;' +
    '  color: #00f0ff;' +
    '  font-family: Consolas, "Cascadia Code", "Fira Code", monospace;' +
    '  overflow: hidden;' +
    '  text-overflow: ellipsis;' +
    '  white-space: nowrap;' +
    '}' +

    /* Status badge */
    '.docker-status-badge {' +
    '  display: inline-block;' +
    '  padding: 2px 8px;' +
    '  border-radius: 10px;' +
    '  font-size: 11px;' +
    '  font-weight: 600;' +
    '  text-transform: uppercase;' +
    '  white-space: nowrap;' +
    '  flex-shrink: 0;' +
    '}' +

    /* Image name */
    '.docker-card-image {' +
    '  font-size: 12px;' +
    '  color: #888;' +
    '  font-family: Consolas, "Cascadia Code", "Fira Code", monospace;' +
    '  overflow: hidden;' +
    '  text-overflow: ellipsis;' +
    '  white-space: nowrap;' +
    '}' +

    /* Ports */
    '.docker-card-ports {' +
    '  display: flex;' +
    '  flex-wrap: wrap;' +
    '  gap: 4px;' +
    '}' +
    '.docker-port {' +
    '  display: inline-block;' +
    '  padding: 1px 6px;' +
    '  border-radius: 3px;' +
    '  font-size: 11px;' +
    '  font-weight: 600;' +
    '  font-family: Consolas, "Cascadia Code", "Fira Code", monospace;' +
    '  background: rgba(255, 204, 0, 0.12);' +
    '  color: #ffcc00;' +
    '}' +

    /* Stats row */
    '.docker-card-stats {' +
    '  display: flex;' +
    '  gap: 16px;' +
    '}' +
    '.docker-stat {' +
    '  display: flex;' +
    '  flex-direction: column;' +
    '  gap: 2px;' +
    '}' +
    '.docker-stat-label {' +
    '  font-size: 10px;' +
    '  color: #888;' +
    '  text-transform: uppercase;' +
    '  letter-spacing: 0.5px;' +
    '}' +
    '.docker-stat-value {' +
    '  font-size: 12px;' +
    '  color: #e0e0e0;' +
    '  font-family: Consolas, "Cascadia Code", "Fira Code", monospace;' +
    '}' +

    /* Uptime */
    '.docker-card-uptime {' +
    '  font-size: 11px;' +
    '  color: #888;' +
    '}' +

    /* Actions */
    '.docker-card-actions {' +
    '  display: flex;' +
    '  gap: 6px;' +
    '  margin-top: 4px;' +
    '}' +

    /* Toast animation */
    '#docker-toast {' +
    '  animation: docker-toast-in 0.2s ease;' +
    '}' +
    '@keyframes docker-toast-in {' +
    '  from { opacity: 0; transform: translateY(8px); }' +
    '  to   { opacity: 1; transform: translateY(0); }' +
    '}';

  document.head.appendChild(style);
}
