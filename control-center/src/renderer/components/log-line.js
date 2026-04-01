// Log line component — creates colored, timestamped log entry

const SERVICE_COLORS = {
  'vite': '#00f0ff',
  'dev-server': '#b060ff',
  'api-server': '#44ff88',
  'docker': '#ffcc00',
  'system': '#666688',
};

function createLogLine(entry) {
  const el = document.createElement('div');
  el.className = 'log-line';

  const time = new Date(entry.timestamp);
  const timeStr = time.toTimeString().slice(0, 8) + '.' +
    String(time.getMilliseconds()).padStart(3, '0');

  const color = SERVICE_COLORS[entry.source] || '#666688';
  const severityClass = entry.severity === 'error' ? 'error' :
    entry.severity === 'warn' ? 'warn' : '';

  el.innerHTML = `
    <span class="log-time">${timeStr}</span>
    <span class="log-source" style="background:${color}22;color:${color}">${entry.source}</span>
    <span class="log-text ${severityClass}">${escapeHtml(entry.text)}</span>
  `;

  return el;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
