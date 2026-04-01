// Service card component — creates DOM element for one service

const STATUS_CLASSES = {
  running: 'running',
  stopped: 'stopped',
  starting: 'starting',
  stopping: 'starting',
  error: 'error',
};

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
    <div class="actions"></div>
  `;

  updateCardActions(card, service);
  updateCardMeta(card, service);
  return card;
}

function updateServiceCard(card, service) {
  const badge = card.querySelector('.status-badge');
  badge.className = `status-badge ${STATUS_CLASSES[service.status] || 'stopped'}`;
  badge.textContent = service.status;
  updateCardActions(card, service);
  updateCardMeta(card, service);
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
    startBtn.onclick = () => window.artifex.startService(service.id);
    actions.appendChild(startBtn);
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
