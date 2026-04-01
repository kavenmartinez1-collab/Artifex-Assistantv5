// Progress bar component for quantization and long-running tasks
// Usage: const bar = createProgressBar(container); bar.update(42, 'Quantizing layer 12/28');

function createProgressBar(container) {
  const wrapper = document.createElement('div');
  wrapper.className = 'qprogress-wrapper';

  const header = document.createElement('div');
  header.className = 'qprogress-header';

  const labelEl = document.createElement('span');
  labelEl.className = 'qprogress-label';
  labelEl.textContent = '';

  const percentEl = document.createElement('span');
  percentEl.className = 'qprogress-percent';
  percentEl.textContent = '0%';

  header.appendChild(labelEl);
  header.appendChild(percentEl);

  const track = document.createElement('div');
  track.className = 'qprogress-track';

  const fill = document.createElement('div');
  fill.className = 'qprogress-fill';
  fill.style.width = '0%';

  track.appendChild(fill);

  const etaEl = document.createElement('div');
  etaEl.className = 'qprogress-eta';
  etaEl.textContent = '';

  wrapper.appendChild(header);
  wrapper.appendChild(track);
  wrapper.appendChild(etaEl);
  container.appendChild(wrapper);

  let startTime = null;

  function update(percent, label, eta) {
    if (startTime === null && percent > 0) startTime = Date.now();

    const clamped = Math.max(0, Math.min(100, percent));
    fill.style.width = clamped + '%';
    percentEl.textContent = Math.round(clamped) + '%';

    if (label !== undefined) {
      labelEl.textContent = label;
    }

    if (eta !== undefined) {
      etaEl.textContent = eta;
    } else if (startTime && clamped > 0 && clamped < 100) {
      // Auto-compute ETA from elapsed time and progress
      const elapsed = (Date.now() - startTime) / 1000;
      const rate = clamped / elapsed;
      const remaining = (100 - clamped) / rate;
      if (remaining < 60) {
        etaEl.textContent = 'ETA: ' + Math.ceil(remaining) + 's';
      } else if (remaining < 3600) {
        const mins = Math.floor(remaining / 60);
        const secs = Math.ceil(remaining % 60);
        etaEl.textContent = 'ETA: ' + mins + 'm ' + secs + 's';
      } else {
        const hrs = Math.floor(remaining / 3600);
        const mins = Math.floor((remaining % 3600) / 60);
        etaEl.textContent = 'ETA: ' + hrs + 'h ' + mins + 'm';
      }
    } else if (clamped >= 100) {
      etaEl.textContent = 'Complete';
    }

    // Completed state
    if (clamped >= 100) {
      fill.classList.add('qprogress-complete');
    } else {
      fill.classList.remove('qprogress-complete');
    }
  }

  function destroy() {
    wrapper.remove();
  }

  // Inject styles once
  if (!document.getElementById('qprogress-styles')) {
    const style = document.createElement('style');
    style.id = 'qprogress-styles';
    style.textContent = `
      .qprogress-wrapper {
        padding: 12px 0;
      }
      .qprogress-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 6px;
      }
      .qprogress-label {
        font-size: 12px;
        color: var(--text);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        flex: 1;
        margin-right: 12px;
      }
      .qprogress-percent {
        font-size: 12px;
        font-weight: 600;
        color: var(--accent);
        font-family: var(--mono);
        flex-shrink: 0;
      }
      .qprogress-track {
        height: 6px;
        background: var(--bg);
        border-radius: 3px;
        overflow: hidden;
      }
      .qprogress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        border-radius: 3px;
        transition: width 0.3s ease;
        min-width: 0;
      }
      .qprogress-fill.qprogress-complete {
        background: linear-gradient(90deg, var(--success), var(--accent));
      }
      .qprogress-eta {
        font-size: 11px;
        color: var(--dim);
        margin-top: 4px;
        min-height: 16px;
        font-family: var(--mono);
      }
    `;
    document.head.appendChild(style);
  }

  return { update: update, destroy: destroy };
}
