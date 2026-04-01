// Toast notification component — bottom-right stacking notifications
// Usage: showToast('Model loaded', 'success');  showToast('Connection lost', 'error', 5000);

const TOAST_TYPE_COLORS = {
  info: 'var(--accent)',
  success: 'var(--success)',
  error: 'var(--error)',
  warning: 'var(--warn)',
};

function showToast(message, type, duration) {
  if (type === undefined) type = 'info';
  if (duration === undefined) duration = 3000;

  // Ensure container exists
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    document.body.appendChild(container);
  }

  const color = TOAST_TYPE_COLORS[type] || TOAST_TYPE_COLORS.info;

  const toast = document.createElement('div');
  toast.className = 'toast-item';
  toast.style.borderLeftColor = color;

  const icon = type === 'success' ? '\u2713' :
    type === 'error' ? '\u2717' :
    type === 'warning' ? '\u26A0' : '\u25CF';

  toast.innerHTML = `
    <span class="toast-icon" style="color:${color}">${icon}</span>
    <span class="toast-text">${escapeToastHtml(message)}</span>
  `;

  container.appendChild(toast);

  // Trigger slide-in on next frame
  requestAnimationFrame(() => {
    toast.classList.add('toast-visible');
  });

  // Auto-dismiss
  const timer = setTimeout(() => dismissToast(toast), duration);

  // Click to dismiss early
  toast.onclick = () => {
    clearTimeout(timer);
    dismissToast(toast);
  };

  // Inject styles once
  if (!document.getElementById('toast-styles')) {
    const style = document.createElement('style');
    style.id = 'toast-styles';
    style.textContent = `
      #toast-container {
        position: fixed;
        bottom: 40px;
        right: 16px;
        display: flex;
        flex-direction: column-reverse;
        gap: 8px;
        z-index: 8000;
        pointer-events: none;
      }
      .toast-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        border-radius: 6px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: var(--text);
        min-width: 240px;
        max-width: 360px;
        pointer-events: auto;
        cursor: pointer;
        transform: translateX(120%);
        opacity: 0;
        transition: transform 0.25s ease, opacity 0.25s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      }
      .toast-item.toast-visible {
        transform: translateX(0);
        opacity: 1;
      }
      .toast-item.toast-dismissing {
        transform: translateX(120%);
        opacity: 0;
      }
      .toast-icon {
        font-size: 14px;
        flex-shrink: 0;
      }
      .toast-text {
        flex: 1;
        line-height: 1.4;
      }
    `;
    document.head.appendChild(style);
  }
}

function dismissToast(toast) {
  toast.classList.remove('toast-visible');
  toast.classList.add('toast-dismissing');
  setTimeout(() => {
    toast.remove();
    // Clean up empty container
    const container = document.getElementById('toast-container');
    if (container && container.children.length === 0) {
      container.remove();
    }
  }, 250);
}

function escapeToastHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
