// Services panel — service cards with start/stop/restart + console output

let selectedServiceId = null;
let serviceCards = {};
let serviceUpdateListener = null;

async function initServicesPanel(container) {
  container.innerHTML = '';

  // Header
  const header = document.createElement('div');
  header.className = 'panel-header';
  header.innerHTML = `
    <h2>Services</h2>
    <div>
      <button class="btn btn-primary btn-sm" id="start-all-btn">Start All</button>
      <button class="btn btn-danger btn-sm" id="stop-all-btn">Stop All</button>
    </div>
  `;
  container.appendChild(header);

  document.getElementById('start-all-btn').onclick = () => window.artifex.startAll();
  document.getElementById('stop-all-btn').onclick = () => window.artifex.stopAll();

  // Card grid
  const grid = document.createElement('div');
  grid.className = 'card-grid';
  grid.id = 'service-grid';
  container.appendChild(grid);

  // Console output
  const consoleSection = document.createElement('div');
  consoleSection.style.padding = '0 16px 16px';
  consoleSection.innerHTML = `
    <div style="font-size:12px;color:var(--dim);margin-bottom:4px">
      Console Output <span id="console-service-name"></span>
    </div>
    <div class="console-output" id="service-console"></div>
  `;
  container.appendChild(consoleSection);

  // Load service options (port, backend, model configs) then services
  await loadServiceOptions();
  const services = await window.artifex.listServices();
  serviceCards = {};
  for (const svc of services) {
    const card = createServiceCard(svc);
    card.addEventListener('click', (e) => {
      if (e.target.tagName === 'BUTTON') return;
      selectService(svc.id);
    });
    grid.appendChild(card);
    serviceCards[svc.id] = card;
  }

  if (services.length > 0) {
    selectService(services[0].id);
  }

  // Listen for updates — store reference for cleanup
  serviceUpdateListener = (status) => {
    const card = serviceCards[status.id];
    if (card) updateServiceCard(card, status);
  };
  window.artifex.onServiceUpdate(serviceUpdateListener);
}

function cleanupServicesPanel() {
  if (serviceUpdateListener && window.artifex) {
    window.artifex.removeServiceListener(serviceUpdateListener);
  }
  serviceUpdateListener = null;
  serviceCards = {};
  selectedServiceId = null;
}

function selectService(id) {
  selectedServiceId = id;
  // Highlight selected card
  Object.values(serviceCards).forEach(c => c.style.borderColor = 'var(--border)');
  if (serviceCards[id]) {
    serviceCards[id].style.borderColor = 'var(--accent)';
  }
  document.getElementById('console-service-name').textContent = `(${id})`;
  // Console output populated by log entries filtered to this service
  const console = document.getElementById('service-console');
  console.textContent = '';
}

// Append log entries to service console if from selected service
function appendToServiceConsole(entry) {
  if (entry.source !== selectedServiceId) return;
  const console = document.getElementById('service-console');
  if (!console) return;
  console.textContent += entry.text + '\n';
  // Auto-scroll
  console.scrollTop = console.scrollHeight;
  // Trim if too long
  const lines = console.textContent.split('\n');
  if (lines.length > 500) {
    console.textContent = lines.slice(-400).join('\n');
  }
}
