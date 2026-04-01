// Mini sparkline graph component for tok/s, VRAM, and other live metrics
// Usage: const graph = createMiniGraph(container, { width: 120, height: 40, color: 'var(--accent)' });
//        graph.addPoint(34.5);

function createMiniGraph(container, opts) {
  const width = (opts && opts.width) || 120;
  const height = (opts && opts.height) || 40;
  const color = (opts && opts.color) || '#00f0ff';
  const maxPoints = (opts && opts.maxPoints) || 60;
  const label = (opts && opts.label) || '';

  const wrapper = document.createElement('div');
  wrapper.className = 'minigraph-wrapper';

  if (label) {
    const labelEl = document.createElement('div');
    labelEl.className = 'minigraph-label';
    labelEl.textContent = label;
    wrapper.appendChild(labelEl);
  }

  const canvasWrap = document.createElement('div');
  canvasWrap.className = 'minigraph-canvas-wrap';

  const canvas = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  canvas.className = 'minigraph-canvas';

  const valueEl = document.createElement('div');
  valueEl.className = 'minigraph-value';
  valueEl.textContent = '--';

  canvasWrap.appendChild(canvas);
  wrapper.appendChild(canvasWrap);
  wrapper.appendChild(valueEl);
  container.appendChild(wrapper);

  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  const points = [];

  // Resolve CSS variable colors to actual hex/rgb for canvas
  function resolveColor(c) {
    if (c.startsWith('var(')) {
      const temp = document.createElement('div');
      temp.style.color = c;
      document.body.appendChild(temp);
      const resolved = getComputedStyle(temp).color;
      temp.remove();
      return resolved;
    }
    return c;
  }

  const resolvedColor = resolveColor(color);

  function draw() {
    ctx.clearRect(0, 0, width, height);

    if (points.length < 2) return;

    // Compute Y range with padding
    let yMin = Infinity;
    let yMax = -Infinity;
    for (let i = 0; i < points.length; i++) {
      if (points[i] < yMin) yMin = points[i];
      if (points[i] > yMax) yMax = points[i];
    }

    // Ensure non-zero range
    if (yMax === yMin) {
      yMax = yMin + 1;
      yMin = Math.max(0, yMin - 1);
    }

    // 10% padding on top and bottom
    const range = yMax - yMin;
    yMin = yMin - range * 0.1;
    yMax = yMax + range * 0.1;

    const pad = 1;
    const drawW = width - pad * 2;
    const drawH = height - pad * 2;
    const step = drawW / (maxPoints - 1);

    function xPos(i) {
      // Align points to right side
      const offset = maxPoints - points.length;
      return pad + (offset + i) * step;
    }

    function yPos(val) {
      return pad + drawH - ((val - yMin) / (yMax - yMin)) * drawH;
    }

    // Filled area
    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(points[0]));
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(xPos(i), yPos(points[i]));
    }
    // Close along bottom
    ctx.lineTo(xPos(points.length - 1), pad + drawH);
    ctx.lineTo(xPos(0), pad + drawH);
    ctx.closePath();

    ctx.fillStyle = resolvedColor.replace(')', ', 0.2)').replace('rgb(', 'rgba(');
    ctx.fill();

    // Line
    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(points[0]));
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(xPos(i), yPos(points[i]));
    }
    ctx.strokeStyle = resolvedColor;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.stroke();

    // Dot on latest point
    const lastX = xPos(points.length - 1);
    const lastY = yPos(points[points.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2, 0, Math.PI * 2);
    ctx.fillStyle = resolvedColor;
    ctx.fill();
  }

  function formatValue(val) {
    if (val >= 1000) return (val / 1000).toFixed(1) + 'k';
    if (val >= 100) return Math.round(val).toString();
    if (val >= 10) return val.toFixed(1);
    return val.toFixed(2);
  }

  function addPoint(value) {
    points.push(value);
    if (points.length > maxPoints) {
      points.shift();
    }
    valueEl.textContent = formatValue(value);
    draw();
  }

  function destroy() {
    wrapper.remove();
  }

  // Inject styles once
  if (!document.getElementById('minigraph-styles')) {
    const style = document.createElement('style');
    style.id = 'minigraph-styles';
    style.textContent = `
      .minigraph-wrapper {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        gap: 2px;
      }
      .minigraph-label {
        font-size: 10px;
        color: var(--dim);
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .minigraph-canvas-wrap {
        border-radius: 4px;
        overflow: hidden;
        background: rgba(15, 17, 26, 0.5);
        border: 1px solid var(--border);
      }
      .minigraph-canvas {
        display: block;
      }
      .minigraph-value {
        font-size: 11px;
        font-family: var(--mono);
        color: var(--text);
        font-weight: 600;
      }
    `;
    document.head.appendChild(style);
  }

  return { addPoint: addPoint, destroy: destroy };
}
