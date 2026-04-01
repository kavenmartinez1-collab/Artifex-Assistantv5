// Wizard step indicator component for multi-step flows (quantize wizard, etc.)
// Usage: const wiz = createWizardSteps(container, ['Model', 'Config', 'Calibrate', 'Export']);
//        wiz.setActive(1);  wiz.setComplete(0);

function createWizardSteps(container, steps) {
  const wrapper = document.createElement('div');
  wrapper.className = 'wizard-steps';

  const nodes = [];

  for (let i = 0; i < steps.length; i++) {
    // Connector line before each step (except the first)
    if (i > 0) {
      const line = document.createElement('div');
      line.className = 'wizard-line';
      wrapper.appendChild(line);
    }

    const stepEl = document.createElement('div');
    stepEl.className = 'wizard-step';

    const circle = document.createElement('div');
    circle.className = 'wizard-circle';
    circle.textContent = i + 1;

    const labelEl = document.createElement('div');
    labelEl.className = 'wizard-step-label';
    labelEl.textContent = steps[i];

    stepEl.appendChild(circle);
    stepEl.appendChild(labelEl);
    wrapper.appendChild(stepEl);

    nodes.push({ el: stepEl, circle: circle });
  }

  container.appendChild(wrapper);

  let activeIndex = 0;

  function render() {
    const lines = wrapper.querySelectorAll('.wizard-line');
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const isComplete = node.el.classList.contains('complete');
      const isActive = i === activeIndex;

      node.el.classList.toggle('active', isActive && !isComplete);
      node.circle.textContent = isComplete ? '\u2713' : (i + 1);
    }

    // Color connector lines based on completion
    for (let i = 0; i < lines.length; i++) {
      const prevComplete = nodes[i].el.classList.contains('complete');
      lines[i].classList.toggle('done', prevComplete);
    }
  }

  function setActive(index) {
    if (index >= 0 && index < nodes.length) {
      activeIndex = index;
      render();
    }
  }

  function setComplete(index) {
    if (index >= 0 && index < nodes.length) {
      nodes[index].el.classList.add('complete');
      render();
    }
  }

  // Initial render
  render();

  // Inject styles once
  if (!document.getElementById('wizard-styles')) {
    const style = document.createElement('style');
    style.id = 'wizard-styles';
    style.textContent = `
      .wizard-steps {
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding: 16px 8px;
        gap: 0;
      }
      .wizard-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        position: relative;
        min-width: 64px;
        flex-shrink: 0;
      }
      .wizard-circle {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        font-family: var(--mono);
        border: 2px solid var(--dim);
        color: var(--dim);
        background: var(--bg);
        transition: all 0.2s ease;
        flex-shrink: 0;
      }
      .wizard-step.active .wizard-circle {
        border-color: var(--accent);
        color: var(--accent);
        box-shadow: 0 0 8px rgba(0, 240, 255, 0.3);
      }
      .wizard-step.complete .wizard-circle {
        border-color: var(--success);
        color: var(--bg);
        background: var(--success);
      }
      .wizard-step-label {
        font-size: 11px;
        color: var(--dim);
        text-align: center;
        white-space: nowrap;
        transition: color 0.2s ease;
      }
      .wizard-step.active .wizard-step-label {
        color: var(--accent);
      }
      .wizard-step.complete .wizard-step-label {
        color: var(--success);
      }
      .wizard-line {
        flex: 1;
        height: 2px;
        background: var(--border);
        margin-top: 14px;
        min-width: 24px;
        transition: background 0.2s ease;
      }
      .wizard-line.done {
        background: var(--success);
      }
    `;
    document.head.appendChild(style);
  }

  return { setActive: setActive, setComplete: setComplete };
}
