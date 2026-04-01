// Modal confirmation dialog component
// Usage: const confirmed = await showModal('Delete Model', 'Are you sure?');

function showModal(title, message, onConfirm, onCancel) {
  return new Promise((resolve) => {
    let resolved = false;

    function close(result) {
      if (resolved) return;
      resolved = true;
      overlay.classList.add('modal-closing');
      setTimeout(() => {
        overlay.remove();
      }, 150);
      if (result && typeof onConfirm === 'function') onConfirm();
      if (!result && typeof onCancel === 'function') onCancel();
      resolve(result);
    }

    // Dark overlay
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    overlay.onclick = (e) => {
      if (e.target === overlay) close(false);
    };

    // Card
    const card = document.createElement('div');
    card.className = 'modal-card';

    const titleEl = document.createElement('div');
    titleEl.className = 'modal-title';
    titleEl.textContent = title;

    const messageEl = document.createElement('div');
    messageEl.className = 'modal-message';
    messageEl.textContent = message;

    const actions = document.createElement('div');
    actions.className = 'modal-actions';

    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'btn btn-sm';
    cancelBtn.textContent = 'Cancel';
    cancelBtn.onclick = () => close(false);

    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'btn btn-danger btn-sm';
    confirmBtn.textContent = 'Confirm';
    confirmBtn.onclick = () => close(true);

    actions.appendChild(cancelBtn);
    actions.appendChild(confirmBtn);

    card.appendChild(titleEl);
    card.appendChild(messageEl);
    card.appendChild(actions);
    overlay.appendChild(card);
    document.body.appendChild(overlay);

    // ESC key closes
    function onKeyDown(e) {
      if (e.key === 'Escape') {
        close(false);
        document.removeEventListener('keydown', onKeyDown);
      }
    }
    document.addEventListener('keydown', onKeyDown);

    // Focus confirm for keyboard accessibility
    confirmBtn.focus();

    // Inject styles once
    if (!document.getElementById('modal-styles')) {
      const style = document.createElement('style');
      style.id = 'modal-styles';
      style.textContent = `
        .modal-overlay {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.6);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 9000;
          animation: modal-fade-in 0.15s ease;
        }
        .modal-overlay.modal-closing {
          opacity: 0;
          transition: opacity 0.15s ease;
        }
        .modal-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 24px;
          min-width: 340px;
          max-width: 480px;
          animation: modal-slide-in 0.15s ease;
        }
        .modal-overlay.modal-closing .modal-card {
          transform: scale(0.95);
          transition: transform 0.15s ease;
        }
        .modal-title {
          font-size: 16px;
          font-weight: 600;
          color: var(--text);
          margin-bottom: 12px;
        }
        .modal-message {
          font-size: 13px;
          color: var(--dim);
          line-height: 1.5;
          margin-bottom: 20px;
        }
        .modal-actions {
          display: flex;
          justify-content: flex-end;
          gap: 8px;
        }
        @keyframes modal-fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes modal-slide-in {
          from { transform: scale(0.95); opacity: 0; }
          to { transform: scale(1); opacity: 1; }
        }
      `;
      document.head.appendChild(style);
    }
  });
}
