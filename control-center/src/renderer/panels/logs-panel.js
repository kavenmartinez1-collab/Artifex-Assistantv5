// Unified log viewer with filtering, auto-scroll, and export

let logContainer = null;
let logEntries = [];
let filterSource = 'all';
let filterSeverity = 'all';
let filterText = '';
let autoScroll = true;
const MAX_VISIBLE = 2000;

async function initLogsPanel(container) {
  container.innerHTML = '';

  // Header
  const header = document.createElement('div');
  header.className = 'panel-header';
  header.innerHTML = `
    <h2>Logs</h2>
    <div>
      <button class="btn btn-sm" id="export-logs-btn">Export</button>
      <button class="btn btn-danger btn-sm" id="clear-logs-btn">Clear</button>
    </div>
  `;
  container.appendChild(header);

  // Filter bar
  const filterBar = document.createElement('div');
  filterBar.className = 'filter-bar';
  filterBar.style.padding = '0 16px';
  filterBar.innerHTML = `
    <select id="filter-source">
      <option value="all">All Services</option>
      <option value="vite">Vite</option>
      <option value="dev-server">Dev Server</option>
      <option value="api-server">API Server</option>
      <option value="docker">Docker</option>
    </select>
    <select id="filter-severity">
      <option value="all">All Levels</option>
      <option value="info">Info</option>
      <option value="warn">Warn</option>
      <option value="error">Error</option>
    </select>
    <input type="text" id="filter-text" placeholder="Search..." style="flex:1" />
  `;
  container.appendChild(filterBar);

  // Log container
  logContainer = document.createElement('div');
  logContainer.id = 'log-container';
  logContainer.style.cssText = 'flex:1;overflow-y:auto;padding:0 16px;';
  container.appendChild(logContainer);

  // Auto-scroll detection
  logContainer.addEventListener('scroll', () => {
    const atBottom = logContainer.scrollTop + logContainer.clientHeight
      >= logContainer.scrollHeight - 20;
    autoScroll = atBottom;
  });

  // Filter handlers
  document.getElementById('filter-source').onchange = (e) => {
    filterSource = e.target.value;
    renderLogs();
  };
  document.getElementById('filter-severity').onchange = (e) => {
    filterSeverity = e.target.value;
    renderLogs();
  };
  document.getElementById('filter-text').oninput = (e) => {
    filterText = e.target.value.toLowerCase();
    renderLogs();
  };

  // Action buttons
  document.getElementById('export-logs-btn').onclick = () => window.artifex.exportLogs();
  document.getElementById('clear-logs-btn').onclick = async () => {
    await window.artifex.clearLogs();
    logEntries = [];
    renderLogs();
  };

  // Load initial buffer
  const buffer = await window.artifex.getLogBuffer(1000);
  logEntries = buffer || [];
  renderLogs();
}

function addLogEntry(entry) {
  logEntries.push(entry);
  // Cap at 10K entries in memory
  if (logEntries.length > 10000) {
    logEntries = logEntries.slice(-8000);
  }

  if (matchesFilter(entry)) {
    appendLogLine(entry);
  }
}

function matchesFilter(entry) {
  if (filterSource !== 'all' && entry.source !== filterSource) return false;
  if (filterSeverity !== 'all' && entry.severity !== filterSeverity) return false;
  if (filterText && !entry.text.toLowerCase().includes(filterText)) return false;
  return true;
}

function renderLogs() {
  if (!logContainer) return;
  logContainer.innerHTML = '';
  const filtered = logEntries.filter(matchesFilter).slice(-MAX_VISIBLE);
  for (const entry of filtered) {
    logContainer.appendChild(createLogLine(entry));
  }
  if (autoScroll) {
    logContainer.scrollTop = logContainer.scrollHeight;
  }
}

function appendLogLine(entry) {
  if (!logContainer) return;
  logContainer.appendChild(createLogLine(entry));
  // Trim DOM if too many
  while (logContainer.children.length > MAX_VISIBLE) {
    logContainer.removeChild(logContainer.firstChild);
  }
  if (autoScroll) {
    logContainer.scrollTop = logContainer.scrollHeight;
  }
}
