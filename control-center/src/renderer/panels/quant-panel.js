// Artifex Control Center — Quantize Wizard Panel
// 6-step wizard: Select Model -> Profile SSM -> Edit Recipe -> Review -> Run -> Done

let wizardState = {
  step: 0,
  selectedModel: null,
  outputName: '',
  recipe: null,
  recipeName: null,
  keepBF16: ['norm', 'embed_tokens'],
  bits: 4,
  groupSize: 128,
  actorder: false,
  klt: false,
  hadamard: false,
  profileData: null,
  isRunning: false,
  progress: null,
};

const STEPS = [
  { label: 'Select Model', icon: '1' },
  { label: 'Profile', icon: '2' },
  { label: 'Recipe', icon: '3' },
  { label: 'Review', icon: '4' },
  { label: 'Quantize', icon: '5' },
  { label: 'Done', icon: '6' },
];

let wizardContainer = null;
let stepIndicator = null;
let stepContent = null;
let progressListener = null;

async function initQuantPanel(container) {
  wizardContainer = container;
  wizardState.step = 0;

  container.innerHTML = `
    <div class="quant-wizard">
      <div class="quant-header">
        <h2>Quantization Wizard</h2>
        <div class="quant-subtitle">Mixed-precision GPTQ quantization for hybrid SSM+attention models</div>
      </div>
      <div id="wizard-steps" class="wizard-steps"></div>
      <div id="wizard-content" class="wizard-content"></div>
      <div id="wizard-nav" class="wizard-nav"></div>
    </div>
  `;

  stepIndicator = container.querySelector('#wizard-steps');
  stepContent = container.querySelector('#wizard-content');

  // Delegated event handler — replaces all inline onclick/onchange
  container.addEventListener('click', _onWizardClick);
  container.addEventListener('change', _onWizardChange);
  container.addEventListener('input', _onWizardInput);

  renderStepIndicator();
  await renderStep();

  // Listen for quantization progress
  if (window.artifex) {
    progressListener = (progress) => {
      wizardState.progress = progress;
      if (wizardState.step === 4) renderStep();
      if (progress.phase === 'done') {
        wizardState.step = 5;
        wizardState.isRunning = false;
        renderStepIndicator();
        renderStep();
      }
      if (progress.phase === 'error') {
        wizardState.isRunning = false;
        renderStep();
      }
    };
    window.artifex.onQuantProgress(progressListener);
  }
}

function cleanupQuantPanel() {
  if (progressListener && window.artifex) {
    window.artifex.removeQuantListener(progressListener);
  }
  if (wizardContainer) {
    wizardContainer.removeEventListener('click', _onWizardClick);
    wizardContainer.removeEventListener('change', _onWizardChange);
    wizardContainer.removeEventListener('input', _onWizardInput);
  }
  progressListener = null;
  wizardContainer = null;
  stepIndicator = null;
  stepContent = null;
}

// ── Delegated event handlers ──

function _onWizardClick(e) {
  const el = e.target.closest('[data-action]');
  if (!el) return;

  const action = el.dataset.action;
  switch (action) {
    case 'select-model':
      selectModel(el.dataset.name, el.dataset.path, el.dataset.quantized === 'true');
      break;
    case 'preset':
      setPreset(el.dataset.preset);
      break;
    case 'wizard-next':
      wizardNext();
      break;
    case 'wizard-back':
      wizardBack();
      break;
    case 'cancel-quant':
      cancelQuant();
      break;
    case 'reset-wizard':
      resetWizard();
      break;
    case 'back-to-review':
      wizardState.step = 3;
      renderStepIndicator();
      renderStep();
      break;
  }
}

function _onWizardChange(e) {
  const el = e.target;
  const field = el.dataset.field;
  if (!field) return;

  switch (field) {
    case 'bits':
      wizardState.bits = parseInt(el.value);
      break;
    case 'group-size':
      wizardState.groupSize = parseInt(el.value);
      break;
    case 'num-samples':
      wizardState.numSamples = parseInt(el.value);
      break;
    case 'bf16-lmhead':
      toggleBF16('lm_head', el.checked);
      break;
  }
}

function _onWizardInput(e) {
  const el = e.target;
  if (el.dataset.field === 'output-name') {
    wizardState.outputName = el.value;
  }
}

// ── Rendering ──

function renderStepIndicator() {
  stepIndicator.innerHTML = STEPS.map((s, i) => {
    const cls = i < wizardState.step ? 'complete'
      : i === wizardState.step ? 'active' : 'upcoming';
    return `<div class="wiz-step ${cls}">
      <div class="wiz-circle">${i < wizardState.step ? '&#10003;' : s.icon}</div>
      <div class="wiz-label">${s.label}</div>
    </div>`;
  }).join('<div class="wiz-line"></div>');
}

async function renderStep() {
  const nav = wizardContainer.querySelector('#wizard-nav');

  switch (wizardState.step) {
    case 0: await renderSelectModel(); break;
    case 1: await renderProfile(); break;
    case 2: renderRecipe(); break;
    case 3: renderReview(); break;
    case 4: renderQuantize(); break;
    case 5: renderDone(); break;
  }

  // Nav buttons
  const canBack = wizardState.step > 0 && wizardState.step < 4 && !wizardState.isRunning;
  const canNext = wizardState.step < 5 && !wizardState.isRunning;
  const nextLabel = wizardState.step === 3 ? 'Start Quantization' : wizardState.step === 4 ? 'Running...' : 'Next';

  nav.innerHTML = `
    ${canBack ? '<button class="btn btn-dim" data-action="wizard-back">Back</button>' : '<span></span>'}
    ${wizardState.step < 5 ? `<button class="btn btn-accent" data-action="wizard-next" ${!canNext ? 'disabled' : ''}>${nextLabel}</button>` : ''}
  `;
}

async function renderSelectModel() {
  stepContent.innerHTML = '<div class="loading">Loading models...</div>';

  let models = [];
  if (window.artifex?.listQuantModels) {
    models = await window.artifex.listQuantModels();
  } else {
    // Fallback: show placeholder
    models = [
      { name: 'qwen3.5-9b', path: 'models/qwen3.5-9b', sizeGB: '19.0 GB', isQuantized: false },
    ];
  }

  const baseModels = models.filter(m => !m.isQuantized);
  const allModels = models;

  stepContent.innerHTML = `
    <h3>Select Source Model</h3>
    <p class="hint">Choose the base (unquantized) model to quantize. BF16 models produce the best results.</p>
    <div class="model-select-grid">
      ${allModels.map(m => `
        <div class="model-select-card ${m.name === wizardState.selectedModel?.name ? 'selected' : ''} ${m.isQuantized ? 'quantized' : ''}"
             data-action="select-model" data-name="${m.name}" data-path="${m.path}" data-quantized="${m.isQuantized}">
          <div class="model-name">${m.name}</div>
          <div class="model-meta">${m.sizeGB} ${m.isQuantized ? '(already quantized)' : '(base BF16)'}</div>
        </div>
      `).join('')}
    </div>
    <div class="field-row" style="margin-top:16px">
      <label>Output name:</label>
      <input type="text" id="output-name" class="field-input"
        data-field="output-name"
        value="${wizardState.outputName || (wizardState.selectedModel ? wizardState.selectedModel.name + '-GPTQ' : '')}"
        placeholder="e.g., qwen3.5-9b-HailMary">
    </div>
  `;
}

function selectModel(name, modelPath, isQuantized) {
  wizardState.selectedModel = { name, path: modelPath, isQuantized };
  wizardState.outputName = name + '-GPTQ';
  renderStep();
}

async function renderProfile() {
  stepContent.innerHTML = `
    <h3>SSM Activation Profiling</h3>
    <p class="hint">Profile the model's SSM layers to determine which projections need higher precision.
    This helps generate an optimal quantization recipe.</p>
    <div class="profile-options">
      <label><input type="checkbox" id="skip-profile" checked> Skip profiling (use recommended defaults for Qwen3.5)</label>
    </div>
    <div class="profile-info">
      <div class="info-card">
        <div class="info-title">Recommended for Qwen3.5-9B:</div>
        <ul>
          <li><strong>BF16</strong>: SSM projections (in_proj_qkv/a/b/z, out_proj) -- recurrence requires full precision</li>
          <li><strong>BF16</strong>: embed_tokens -- errors compound through 24 SSM layers</li>
          <li><strong>INT4 GPTQ</strong>: attention projections (q/k/v/o_proj) -- 8 layers, no recurrence</li>
          <li><strong>INT4 GPTQ</strong>: FFN (gate/up/down_proj) -- 32 layers, lowest sensitivity</li>
          <li><strong>INT4 RTN</strong>: lm_head -- saves 1.4 GB, works without KLT rotation</li>
        </ul>
        <div class="info-highlight">Target: ~5.74 GB (fits 8 GB VRAM with 2+ GB headroom)</div>
      </div>
    </div>
  `;
}

function renderRecipe() {
  stepContent.innerHTML = `
    <h3>Quantization Recipe</h3>
    <p class="hint">Configure per-layer precision. The defaults are proven to produce coherent output on 8 GB VRAM.</p>

    <div class="recipe-section">
      <h4>Preset</h4>
      <div class="preset-btns">
        <button class="btn ${wizardState.keepBF16.includes('lm_head') ? '' : 'btn-accent'}"
          data-action="preset" data-preset="hailmary">HailMary (5.7 GB)</button>
        <button class="btn ${wizardState.keepBF16.includes('lm_head') ? 'btn-accent' : ''}"
          data-action="preset" data-preset="noact">Conservative (9.4 GB)</button>
      </div>
    </div>

    <div class="recipe-section">
      <h4>Keep at BF16 (not quantized)</h4>
      <div class="bf16-checks">
        <label><input type="checkbox" checked disabled> norm weights (always BF16)</label>
        <label><input type="checkbox" checked disabled> embed_tokens (always BF16 -- SSM recurrence)</label>
        <label><input type="checkbox" id="bf16-ssm" checked disabled> SSM projections (always BF16 -- recurrence)</label>
        <label><input type="checkbox" id="bf16-lmhead" data-field="bf16-lmhead" ${wizardState.keepBF16.includes('lm_head') ? 'checked' : ''}> lm_head (BF16 = higher quality, +1.4 GB)</label>
      </div>
    </div>

    <div class="recipe-section">
      <h4>Quantization Settings</h4>
      <div class="field-row">
        <label>Bits:</label>
        <select class="field-input" data-field="bits">
          <option value="4" ${wizardState.bits === 4 ? 'selected' : ''}>INT4 (recommended)</option>
          <option value="8" ${wizardState.bits === 8 ? 'selected' : ''}>INT8 (higher quality, larger)</option>
        </select>
      </div>
      <div class="field-row">
        <label>Group size:</label>
        <select class="field-input" data-field="group-size">
          <option value="128" ${wizardState.groupSize === 128 ? 'selected' : ''}>128 (recommended)</option>
          <option value="64" ${wizardState.groupSize === 64 ? 'selected' : ''}>64 (slightly better quality)</option>
          <option value="32" ${wizardState.groupSize === 32 ? 'selected' : ''}>32 (best quality, larger)</option>
        </select>
      </div>
      <div class="field-row">
        <label>Calibration samples:</label>
        <input type="number" class="field-input" data-field="num-samples" value="128" min="16" max="512" style="width:80px">
      </div>
    </div>

    ${wizardState.recipeName ? `<div class="recipe-file">Using recipe: ${wizardState.recipeName}</div>` : ''}
  `;
}

function setPreset(preset) {
  if (preset === 'hailmary') {
    wizardState.keepBF16 = ['norm', 'embed_tokens'];
    wizardState.bits = 4;
    wizardState.groupSize = 128;
    wizardState.actorder = false;
  } else if (preset === 'noact') {
    wizardState.keepBF16 = ['norm', 'embed_tokens', 'lm_head'];
    wizardState.bits = 4;
    wizardState.groupSize = 128;
    wizardState.actorder = false;
  }
  renderStep();
}

function toggleBF16(pattern, checked) {
  if (checked && !wizardState.keepBF16.includes(pattern)) {
    wizardState.keepBF16.push(pattern);
  } else if (!checked) {
    wizardState.keepBF16 = wizardState.keepBF16.filter(p => p !== pattern);
  }
}

function renderReview() {
  const model = wizardState.selectedModel;
  const outputName = wizardState.outputName || (model ? model.name + '-GPTQ' : 'output');

  // Estimate output size
  let estimatedSize = '~5.7 GB';
  if (wizardState.keepBF16.includes('lm_head')) estimatedSize = '~9.4 GB';
  if (wizardState.bits === 8) estimatedSize = '~7.9 GB';

  stepContent.innerHTML = `
    <h3>Review Configuration</h3>
    <div class="review-card">
      <div class="review-row"><span class="review-label">Source model:</span> <span>${model?.name || 'none'}</span></div>
      <div class="review-row"><span class="review-label">Output:</span> <span>models/${outputName}</span></div>
      <div class="review-row"><span class="review-label">Bits:</span> <span>INT${wizardState.bits}</span></div>
      <div class="review-row"><span class="review-label">Group size:</span> <span>${wizardState.groupSize}</span></div>
      <div class="review-row"><span class="review-label">BF16 preserved:</span> <span>${wizardState.keepBF16.join(', ')}</span></div>
      <div class="review-row"><span class="review-label">SSM layers:</span> <span>BF16 (always -- recurrence)</span></div>
      <div class="review-row"><span class="review-label">Calibration:</span> <span>${wizardState.numSamples || 128} samples, wikitext</span></div>
      <div class="review-row"><span class="review-label">Estimated size:</span> <span class="accent">${estimatedSize}</span></div>
      <div class="review-row"><span class="review-label">Estimated time:</span> <span>~28 minutes on CUDA GPU</span></div>
    </div>
    <div class="review-warning">
      <strong>Important:</strong> SSM projections (linear_attn.*) are always kept at BF16.
      INT4/INT8 quantization of SSM layers produces gibberish due to DeltaNet recurrence error amplification.
    </div>
  `;
}

function renderQuantize() {
  const p = wizardState.progress;

  if (!p || p.phase === 'error') {
    const errMsg = p?.message || 'Unknown error';
    stepContent.innerHTML = `
      <h3>Quantization ${p?.phase === 'error' ? 'Failed' : 'Running'}</h3>
      ${p?.phase === 'error' ? `
        <div class="error-box">
          <div class="error-title">Error</div>
          <div class="error-msg">${errMsg}</div>
        </div>
        <button class="btn btn-accent" data-action="back-to-review">Back to Review</button>
      ` : `
        <div class="quant-status">Starting quantization...</div>
        <div class="progress-bar-container">
          <div class="progress-bar" style="width: 0%"></div>
        </div>
      `}
    `;
    return;
  }

  const percent = p.percent >= 0 ? p.percent : 0;
  const phaseLabels = {
    loading: 'Loading model...',
    calibrating: 'Computing calibration data...',
    quantizing: `Quantizing layer ${p.layer}/${p.totalLayers}`,
    saving: 'Saving quantized model...',
    profiling: 'Profiling SSM activations...',
    done: 'Complete!',
  };

  stepContent.innerHTML = `
    <h3>Quantizing...</h3>
    <div class="quant-status">${phaseLabels[p.phase] || p.message}</div>
    <div class="progress-bar-container">
      <div class="progress-bar" style="width: ${percent}%"></div>
    </div>
    <div class="progress-meta">
      <span>${percent}%</span>
      ${p.eta ? `<span>ETA: ${p.eta}</span>` : ''}
    </div>
    ${p.sublayer ? `<div class="quant-sublayer">${p.sublayer}</div>` : ''}
    <button class="btn btn-error" data-action="cancel-quant" style="margin-top:16px">Cancel</button>
  `;
}

function renderDone() {
  const outputName = wizardState.outputName || 'output';
  stepContent.innerHTML = `
    <h3>Quantization Complete!</h3>
    <div class="done-card">
      <div class="done-icon">&#10003;</div>
      <div class="done-msg">Model saved to <strong>models/${outputName}</strong></div>
      <div class="done-hint">Load it in the WebGPU engine by selecting it from the model browser.</div>
    </div>
    <button class="btn btn-accent" data-action="reset-wizard">Quantize Another Model</button>
  `;
}

async function wizardNext() {
  if (wizardState.step === 0 && !wizardState.selectedModel) {
    if (typeof showToast === 'function') showToast('Please select a model first', 'warning');
    return;
  }

  if (wizardState.step === 3) {
    // Start quantization
    wizardState.isRunning = true;
    wizardState.step = 4;
    renderStepIndicator();
    renderStep();

    if (window.artifex?.runQuantization) {
      try {
        await window.artifex.runQuantization({
          modelPath: wizardState.selectedModel.path,
          outputPath: `models/${wizardState.outputName || wizardState.selectedModel.name + '-GPTQ'}`,
          keepBF16: wizardState.keepBF16,
          bits: wizardState.bits,
          groupSize: wizardState.groupSize,
          numSamples: wizardState.numSamples || 128,
          seqLength: 2048,
          actorder: wizardState.actorder,
          hadamard: wizardState.hadamard,
          klt: wizardState.klt,
          recipe: wizardState.recipe,
          profileSSM: false,
        });
      } catch (err) {
        wizardState.progress = { phase: 'error', message: err.message, layer: 0, totalLayers: 32, sublayer: '', percent: 0, eta: '' };
        wizardState.isRunning = false;
        renderStep();
      }
    }
    return;
  }

  wizardState.step = Math.min(wizardState.step + 1, 5);
  renderStepIndicator();
  await renderStep();
}

function wizardBack() {
  wizardState.step = Math.max(wizardState.step - 1, 0);
  renderStepIndicator();
  renderStep();
}

function cancelQuant() {
  if (window.artifex?.cancelQuantization) {
    window.artifex.cancelQuantization();
  }
  wizardState.isRunning = false;
  wizardState.step = 3;
  renderStepIndicator();
  renderStep();
}

function resetWizard() {
  wizardState = {
    step: 0, selectedModel: null, outputName: '', recipe: null, recipeName: null,
    keepBF16: ['norm', 'embed_tokens'], bits: 4, groupSize: 128,
    actorder: false, klt: false, hadamard: false, profileData: null,
    isRunning: false, progress: null,
  };
  renderStepIndicator();
  renderStep();
}
