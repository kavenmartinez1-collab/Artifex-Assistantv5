// Artifex Control Center — Renderer entry point
// Panel routing, IPC bridge setup, status bar updates

const panels = {
  services: { init: initServicesPanel, label: 'Services' },
  logs: { init: initLogsPanel, label: 'Logs' },
  quantize: { init: initQuantPanel, label: 'Quantize' },
  cluster: { init: initClusterPanel, label: 'Cluster' },
  models: { init: initModelsPanel, label: 'Models' },
  docker: { init: initDockerPanel, label: 'Docker' },
};

let activePanel = 'services';
let content = null;

document.addEventListener('DOMContentLoaded', async () => {
  content = document.getElementById('content');
  const sidebar = document.getElementById('sidebar');
  const statusbar = document.getElementById('statusbar');

  // Create sidebar buttons
  for (const [id, panel] of Object.entries(panels)) {
    const btn = document.createElement('button');
    btn.className = 'sidebar-btn' + (id === activePanel ? ' active' : '');
    btn.textContent = panel.label;
    btn.dataset.panel = id;
    btn.onclick = () => switchPanel(id);
    sidebar.appendChild(btn);
  }

  // Global IPC listeners
  if (window.artifex) {
    window.artifex.onLogEntry((entry) => {
      // Feed to logs panel
      if (typeof addLogEntry === 'function') addLogEntry(entry);
      // Feed to service console
      if (typeof appendToServiceConsole === 'function') appendToServiceConsole(entry);
    });

    window.artifex.onServiceUpdate((status) => {
      updateStatusBar();
    });
  }

  // Initialize default panel
  await switchPanel(activePanel);
  updateStatusBar();

  // Periodic status bar refresh
  setInterval(updateStatusBar, 5000);
});

async function switchPanel(id) {
  activePanel = id;

  // Update sidebar active state
  document.querySelectorAll('.sidebar-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.panel === id);
  });

  // Clear and initialize panel
  content.innerHTML = '';

  const panel = panels[id];
  if (panel && panel.init) {
    await panel.init(content);
  } else {
    content.innerHTML = `
      <div style="padding:40px;text-align:center;color:var(--dim)">
        <div style="font-size:24px;margin-bottom:8px">${panel ? panel.label : id}</div>
        <div>Coming soon</div>
      </div>
    `;
  }
}

async function updateStatusBar() {
  const statusbar = document.getElementById('statusbar');
  if (!statusbar || !window.artifex) return;

  try {
    const services = await window.artifex.listServices();
    const running = services.filter(s => s.status === 'running').length;
    const total = services.length;
    statusbar.textContent = `Services: ${running}/${total} running`;
  } catch {
    statusbar.textContent = 'Connecting...';
  }
}
