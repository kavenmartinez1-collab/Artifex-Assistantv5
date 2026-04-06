// Artifex Control Center — Renderer entry point
// Panel routing, IPC bridge setup, status bar updates

const panels = {
  services: { init: initServicesPanel, cleanup: cleanupServicesPanel, label: 'Services' },
  logs: { init: initLogsPanel, cleanup: cleanupLogsPanel, label: 'Logs' },
  quantize: { init: initQuantPanel, cleanup: cleanupQuantPanel, label: 'Quantize' },
  cluster: { init: initClusterPanel, cleanup: cleanupClusterPanel, label: 'Cluster' },
  models: { init: initModelsPanel, cleanup: cleanupModelsPanel, label: 'Models' },
  docker: { init: initDockerPanel, cleanup: cleanupDockerPanel, label: 'Docker' },
};

let activePanel = 'services';
let activePanelInitialized = false;
let content = null;

document.addEventListener('DOMContentLoaded', async () => {
  content = document.getElementById('content');
  const sidebar = document.getElementById('sidebar');
  const statusbar = document.getElementById('statusbar');

  // Wire up sidebar buttons (defined in index.html with icons)
  sidebar.querySelectorAll('.sidebar-btn[data-panel]').forEach(btn => {
    btn.onclick = () => switchPanel(btn.dataset.panel);
  });

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
  // Clean up the previous panel before switching (skip on first init)
  if (activePanelInitialized) {
    const prevPanel = panels[activePanel];
    if (prevPanel && prevPanel.cleanup) {
      try { prevPanel.cleanup(); } catch (e) { console.warn('Panel cleanup error:', e); }
    }
  }

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
    activePanelInitialized = true;
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
