// GPU Cluster panel — worker cards, task queue, and connection status
// Requires: mini-graph.js loaded before this file

let clusterWorkerCards = {};
let clusterTaskRows = {};
let clusterSparklines = {};
let clusterUpdateListener = null;
let clusterPollTimer = null;

function initClusterPanel(container) {
  // Clean up any previous instance
  cleanupClusterPanel();

  container.innerHTML = '';

  // Inject cluster-specific styles
  injectClusterStyles();

  // ── Top bar ──
  const topBar = document.createElement('div');
  topBar.className = 'panel-header';
  topBar.innerHTML = `
    <div style="display:flex;align-items:center;gap:12px">
      <h2>GPU Cluster</h2>
      <span class="cluster-conn-dot" id="cluster-conn-dot"></span>
      <span class="cluster-conn-label" id="cluster-conn-label">Connecting...</span>
    </div>
    <div class="cluster-hub-url" id="cluster-hub-url">ws://127.0.0.1:3001/ws</div>
  `;
  container.appendChild(topBar);

  // ── Worker cards section ──
  const workersHeader = document.createElement('div');
  workersHeader.className = 'cluster-section-header';
  workersHeader.textContent = 'Workers';
  container.appendChild(workersHeader);

  const workerGrid = document.createElement('div');
  workerGrid.className = 'card-grid';
  workerGrid.id = 'cluster-worker-grid';
  container.appendChild(workerGrid);

  // ── Empty state for workers ──
  const emptyState = document.createElement('div');
  emptyState.className = 'cluster-empty';
  emptyState.id = 'cluster-empty-state';
  emptyState.textContent = 'No workers connected. Start the dev-server and a GPU worker to see cluster status.';
  container.appendChild(emptyState);

  // ── Task queue section ──
  const tasksHeader = document.createElement('div');
  tasksHeader.className = 'cluster-section-header';
  tasksHeader.textContent = 'Task Queue';
  container.appendChild(tasksHeader);

  const tableWrap = document.createElement('div');
  tableWrap.className = 'cluster-table-wrap';

  const table = document.createElement('table');
  table.className = 'cluster-task-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>ID</th>
        <th>Status</th>
        <th>Worker</th>
        <th>Model</th>
        <th>Tokens</th>
        <th>Speed</th>
        <th>Created</th>
      </tr>
    </thead>
    <tbody id="cluster-task-tbody"></tbody>
  `;
  tableWrap.appendChild(table);
  container.appendChild(tableWrap);

  const emptyTasks = document.createElement('div');
  emptyTasks.className = 'cluster-empty-tasks';
  emptyTasks.id = 'cluster-empty-tasks';
  emptyTasks.textContent = 'No tasks in queue.';
  container.appendChild(emptyTasks);

  // ── Initialize state and listen for updates ──
  clusterWorkerCards = {};
  clusterTaskRows = {};
  clusterSparklines = {};

  // Listen for cluster status updates from main process
  if (window.artifex && window.artifex.onClusterStatus) {
    clusterUpdateListener = function (status) {
      if (status && status.workers !== undefined && status.tasks !== undefined) {
        renderClusterState(status.workers, status.tasks, status.connected, status.timeSeries);
      }
    };
    window.artifex.onClusterStatus(clusterUpdateListener);
  }
}

function cleanupClusterPanel() {
  // Remove IPC listener
  if (clusterUpdateListener && window.artifex && window.artifex.removeClusterListener) {
    window.artifex.removeClusterListener(clusterUpdateListener);
  }
  clusterUpdateListener = null;

  if (clusterPollTimer) {
    clearInterval(clusterPollTimer);
    clusterPollTimer = null;
  }

  // Destroy sparklines
  for (var key in clusterSparklines) {
    if (clusterSparklines[key] && clusterSparklines[key].destroy) {
      clusterSparklines[key].destroy();
    }
  }
  clusterSparklines = {};
  clusterWorkerCards = {};
  clusterTaskRows = {};
}

function renderClusterState(workers, tasks, connected, timeSeries) {
  // ── Connection indicator ──
  var dot = document.getElementById('cluster-conn-dot');
  var label = document.getElementById('cluster-conn-label');
  if (dot && label) {
    if (connected) {
      dot.className = 'cluster-conn-dot connected';
      label.textContent = 'Connected';
      label.style.color = 'var(--success)';
    } else {
      dot.className = 'cluster-conn-dot disconnected';
      label.textContent = 'Disconnected';
      label.style.color = 'var(--error)';
    }
  }

  // ── Workers ──
  var grid = document.getElementById('cluster-worker-grid');
  var emptyState = document.getElementById('cluster-empty-state');
  if (!grid) return;

  if (!workers || workers.length === 0) {
    grid.style.display = 'none';
    if (emptyState) emptyState.style.display = 'block';
  } else {
    grid.style.display = 'flex';
    if (emptyState) emptyState.style.display = 'none';
  }

  // Track which workers are still present
  var activeIds = {};

  for (var i = 0; i < workers.length; i++) {
    var w = workers[i];
    activeIds[w.id] = true;

    if (clusterWorkerCards[w.id]) {
      updateWorkerCard(clusterWorkerCards[w.id], w, timeSeries);
    } else {
      var card = createWorkerCard(w, timeSeries);
      grid.appendChild(card);
      clusterWorkerCards[w.id] = card;
    }
  }

  // Remove cards for workers that are gone
  for (var wid in clusterWorkerCards) {
    if (!activeIds[wid]) {
      clusterWorkerCards[wid].remove();
      delete clusterWorkerCards[wid];
      if (clusterSparklines[wid]) {
        clusterSparklines[wid].destroy();
        delete clusterSparklines[wid];
      }
    }
  }

  // ── Tasks ──
  var tbody = document.getElementById('cluster-task-tbody');
  var emptyTasks = document.getElementById('cluster-empty-tasks');
  if (!tbody) return;

  if (!tasks || tasks.length === 0) {
    tbody.innerHTML = '';
    if (emptyTasks) emptyTasks.style.display = 'block';
  } else {
    if (emptyTasks) emptyTasks.style.display = 'none';
  }

  // Rebuild task rows (tasks change order/status frequently, simpler to rebuild)
  tbody.innerHTML = '';
  clusterTaskRows = {};

  if (tasks && tasks.length > 0) {
    // Sort: running first, then queued, then completed/failed (newest first)
    var statusOrder = { running: 0, queued: 1, completed: 2, failed: 3 };
    var sorted = tasks.slice().sort(function (a, b) {
      var oa = statusOrder[a.status] !== undefined ? statusOrder[a.status] : 4;
      var ob = statusOrder[b.status] !== undefined ? statusOrder[b.status] : 4;
      if (oa !== ob) return oa - ob;
      return b.created - a.created;
    });

    for (var j = 0; j < sorted.length; j++) {
      var row = createTaskRow(sorted[j]);
      tbody.appendChild(row);
      clusterTaskRows[sorted[j].id] = row;
    }
  }
}

// ── Worker Card ──

function createWorkerCard(worker, timeSeries) {
  var card = document.createElement('div');
  card.className = 'service-card cluster-worker-card';
  card.dataset.id = worker.id;

  // GPU name
  var nameEl = document.createElement('div');
  nameEl.className = 'name';
  nameEl.textContent = worker.gpu || worker.id;
  card.appendChild(nameEl);

  // Worker ID
  var idEl = document.createElement('div');
  idEl.className = 'port';
  idEl.textContent = worker.id;
  card.appendChild(idEl);

  // Status badge
  var badgeWrap = document.createElement('div');
  var badge = document.createElement('span');
  badge.className = 'status-badge ' + workerStatusClass(worker.status);
  badge.textContent = worker.status;
  badgeWrap.appendChild(badge);
  card.appendChild(badgeWrap);

  // VRAM bar
  var vramWrap = document.createElement('div');
  vramWrap.className = 'cluster-vram-section';

  var vramLabel = document.createElement('div');
  vramLabel.className = 'cluster-vram-label';
  vramLabel.innerHTML = '<span>VRAM</span><span class="cluster-vram-text"></span>';
  vramWrap.appendChild(vramLabel);

  var vramTrack = document.createElement('div');
  vramTrack.className = 'cluster-vram-track';
  var vramFill = document.createElement('div');
  vramFill.className = 'cluster-vram-fill';
  vramTrack.appendChild(vramFill);
  vramWrap.appendChild(vramTrack);
  card.appendChild(vramWrap);

  // Current task
  var taskEl = document.createElement('div');
  taskEl.className = 'meta cluster-current-task';
  card.appendChild(taskEl);

  // tok/s value
  var toksEl = document.createElement('div');
  toksEl.className = 'cluster-toks-value';
  card.appendChild(toksEl);

  // Sparkline container
  var sparkWrap = document.createElement('div');
  sparkWrap.className = 'cluster-spark-wrap';
  card.appendChild(sparkWrap);

  // Create sparkline graph
  var spark = createMiniGraph(sparkWrap, {
    width: 160,
    height: 36,
    color: 'var(--accent)',
    maxPoints: 60,
    label: 'tok/s',
  });
  clusterSparklines[worker.id] = spark;

  // Populate initial sparkline data from time series
  if (timeSeries && timeSeries[worker.id] && timeSeries[worker.id].tokPerSec) {
    var pts = timeSeries[worker.id].tokPerSec;
    for (var k = 0; k < pts.length; k++) {
      spark.addPoint(pts[k]);
    }
  }

  // Set initial values
  updateWorkerCardValues(card, worker);

  return card;
}

function updateWorkerCard(card, worker, timeSeries) {
  updateWorkerCardValues(card, worker);

  // Update sparkline with latest tok/s
  if (clusterSparklines[worker.id] && worker.tokPerSec !== undefined) {
    clusterSparklines[worker.id].addPoint(worker.tokPerSec);
  }
}

function updateWorkerCardValues(card, worker) {
  // GPU name
  var nameEl = card.querySelector('.name');
  if (nameEl) nameEl.textContent = worker.gpu || worker.id;

  // Status badge
  var badge = card.querySelector('.status-badge');
  if (badge) {
    badge.className = 'status-badge ' + workerStatusClass(worker.status);
    badge.textContent = worker.status;
  }

  // VRAM bar
  var vramText = card.querySelector('.cluster-vram-text');
  var vramFill = card.querySelector('.cluster-vram-fill');
  if (vramText && vramFill) {
    var usedMB = worker.vramUsed || 0;
    var totalMB = worker.vram || 1;
    var pct = Math.min(100, (usedMB / totalMB) * 100);
    vramText.textContent = formatMB(usedMB) + ' / ' + formatMB(totalMB);
    vramFill.style.width = pct + '%';

    // Color based on utilization
    if (pct >= 90) {
      vramFill.style.background = 'var(--error)';
    } else if (pct >= 70) {
      vramFill.style.background = 'var(--warning)';
    } else {
      vramFill.style.background = 'var(--success)';
    }
  }

  // Current task
  var taskEl = card.querySelector('.cluster-current-task');
  if (taskEl) {
    if (worker.currentTask) {
      taskEl.textContent = 'Task: ' + worker.currentTask;
      taskEl.style.color = 'var(--text)';
    } else {
      taskEl.textContent = 'No active task';
      taskEl.style.color = 'var(--dim)';
    }
  }

  // tok/s
  var toksEl = card.querySelector('.cluster-toks-value');
  if (toksEl) {
    var tps = worker.tokPerSec || 0;
    toksEl.textContent = tps > 0 ? tps.toFixed(1) + ' tok/s' : '-- tok/s';
  }
}

function workerStatusClass(status) {
  switch (status) {
    case 'idle': return 'stopped';
    case 'busy': return 'running';
    case 'offline': return 'error';
    default: return 'stopped';
  }
}

// ── Task Row ──

function createTaskRow(task) {
  var tr = document.createElement('tr');
  tr.className = 'cluster-task-row';
  tr.dataset.id = task.id;

  // ID (truncated)
  var tdId = document.createElement('td');
  tdId.className = 'cluster-td-id';
  tdId.textContent = truncateId(task.id);
  tdId.title = task.id;
  tr.appendChild(tdId);

  // Status badge
  var tdStatus = document.createElement('td');
  var statusBadge = document.createElement('span');
  statusBadge.className = 'status-badge ' + taskStatusClass(task.status);
  statusBadge.textContent = task.status;
  tdStatus.appendChild(statusBadge);
  tr.appendChild(tdStatus);

  // Worker
  var tdWorker = document.createElement('td');
  tdWorker.className = 'cluster-td-worker';
  tdWorker.textContent = task.worker ? truncateId(task.worker) : '--';
  tdWorker.title = task.worker || '';
  tr.appendChild(tdWorker);

  // Model
  var tdModel = document.createElement('td');
  tdModel.textContent = task.model || '--';
  tr.appendChild(tdModel);

  // Tokens
  var tdTokens = document.createElement('td');
  tdTokens.className = 'cluster-td-mono';
  tdTokens.textContent = task.tokens > 0 ? task.tokens.toLocaleString() : '--';
  tr.appendChild(tdTokens);

  // Speed
  var tdSpeed = document.createElement('td');
  tdSpeed.className = 'cluster-td-mono';
  tdSpeed.textContent = task.speed > 0 ? task.speed.toFixed(1) + ' t/s' : '--';
  tr.appendChild(tdSpeed);

  // Created
  var tdCreated = document.createElement('td');
  tdCreated.className = 'cluster-td-time';
  tdCreated.textContent = task.created ? formatTimestamp(task.created) : '--';
  tr.appendChild(tdCreated);

  return tr;
}

function taskStatusClass(status) {
  switch (status) {
    case 'queued': return 'stopped';       // dim
    case 'running': return 'starting';     // accent
    case 'completed': return 'running';    // green
    case 'failed': return 'error';         // red
    default: return 'stopped';
  }
}

// ── Helpers ──

function formatMB(mb) {
  if (mb >= 1024) {
    return (mb / 1024).toFixed(1) + ' GB';
  }
  return Math.round(mb) + ' MB';
}

function truncateId(id) {
  if (!id) return '--';
  if (id.length <= 12) return id;
  return id.substring(0, 10) + '...';
}

function formatTimestamp(ts) {
  var d = new Date(ts);
  var h = d.getHours().toString().padStart(2, '0');
  var m = d.getMinutes().toString().padStart(2, '0');
  var s = d.getSeconds().toString().padStart(2, '0');
  return h + ':' + m + ':' + s;
}

// ── Styles ──

function injectClusterStyles() {
  if (document.getElementById('cluster-panel-styles')) return;

  var style = document.createElement('style');
  style.id = 'cluster-panel-styles';
  style.textContent = `
    /* Connection indicator */
    .cluster-conn-dot {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--dim);
      flex-shrink: 0;
    }
    .cluster-conn-dot.connected {
      background: var(--success);
      box-shadow: 0 0 6px var(--success);
    }
    .cluster-conn-dot.disconnected {
      background: var(--error);
      box-shadow: 0 0 6px var(--error);
    }
    .cluster-conn-label {
      font-size: 12px;
      font-family: var(--mono);
      color: var(--dim);
    }
    .cluster-hub-url {
      font-size: 11px;
      font-family: var(--mono);
      color: var(--dim);
      padding: 2px 8px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 4px;
    }

    /* Section headers */
    .cluster-section-header {
      font-size: 12px;
      font-weight: 600;
      color: var(--dim);
      text-transform: uppercase;
      letter-spacing: 1px;
      padding: 12px 16px 4px;
    }

    /* Empty states */
    .cluster-empty {
      padding: 32px 16px;
      text-align: center;
      color: var(--dim);
      font-size: 13px;
      font-style: italic;
      line-height: 1.6;
    }
    .cluster-empty-tasks {
      padding: 16px;
      text-align: center;
      color: var(--dim);
      font-size: 12px;
      font-style: italic;
    }

    /* Worker card extensions */
    .cluster-worker-card {
      min-width: 240px;
      max-width: 320px;
      flex: 1 0 240px;
    }

    /* VRAM bar */
    .cluster-vram-section {
      margin-top: 4px;
    }
    .cluster-vram-label {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      font-size: 11px;
      color: var(--dim);
      margin-bottom: 3px;
    }
    .cluster-vram-text {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--text);
    }
    .cluster-vram-track {
      height: 6px;
      background: var(--bg);
      border-radius: 3px;
      overflow: hidden;
    }
    .cluster-vram-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 0.3s ease, background 0.3s ease;
      min-width: 0;
      background: var(--success);
    }

    /* Current task */
    .cluster-current-task {
      font-size: 11px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    /* tok/s value */
    .cluster-toks-value {
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 600;
      color: var(--accent);
      margin-top: 2px;
    }

    /* Sparkline container */
    .cluster-spark-wrap {
      margin-top: 4px;
    }

    /* Task table */
    .cluster-table-wrap {
      overflow-x: auto;
      overflow-y: auto;
      max-height: 300px;
      margin: 0 16px 16px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
    }
    .cluster-task-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .cluster-task-table thead {
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .cluster-task-table th {
      text-align: left;
      padding: 8px 10px;
      background: var(--bg2);
      color: var(--dim);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }
    .cluster-task-table td {
      padding: 6px 10px;
      border-bottom: 1px solid rgba(34, 34, 68, 0.3);
      color: var(--text);
      white-space: nowrap;
    }
    .cluster-task-row:hover {
      background: rgba(0, 240, 255, 0.03);
    }
    .cluster-td-id {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--dim);
      max-width: 100px;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .cluster-td-worker {
      font-family: var(--mono);
      font-size: 11px;
    }
    .cluster-td-mono {
      font-family: var(--mono);
    }
    .cluster-td-time {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--dim);
    }

    /* Warning color variable for VRAM */
    :root {
      --warning: #ff9800;
    }
  `;
  document.head.appendChild(style);
}
