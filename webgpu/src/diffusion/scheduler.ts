// FLUX.2-klein FlowMatchEulerDiscrete schedule + Euler step (CPU).
//
// Replicates the exact float32 rounding of the Python reference so the
// parity gate can demand ≤1e-12 vs the fixture:
//   - pipeline: sigmas = np.linspace(1, 1/N, N), mu = compute_empirical_mu
//   - scheduler.set_timesteps: sigmas.astype(float32), then exponential
//     dynamic shift sigma' = e^mu / (e^mu + (1/sigma - 1)) computed
//     elementwise in f32 (numpy f32-array ⊕ python-scalar stays f32),
//     terminal 0 appended; timesteps = sigmas * 1000 (f32)
//   - scheduler.step: prev = sample + (sigma_next - sigma) * model_output,
//     all f32
// (venv diffusers 0.39 scheduling_flow_match_euler_discrete.py + the
//  compute_empirical_mu constants from pipeline_flux2_klein.py:63-78.)

const f32 = Math.fround;

/** pipeline_flux2_klein.compute_empirical_mu, verbatim (f64). */
export function computeEmpiricalMu(imageSeqLen: number, numSteps: number): number {
  const a1 = 8.73809524e-05, b1 = 1.89833333;
  const a2 = 0.00016927, b2 = 0.45666666;
  if (imageSeqLen > 4300) return a2 * imageSeqLen + b2;
  const m200 = a2 * imageSeqLen + b2;
  const m10 = a1 * imageSeqLen + b1;
  const a = (m200 - m10) / 190.0;
  const b = m200 - 200.0 * a;
  return a * numSteps + b;
}

export interface Flux2Schedule {
  mu: number;
  /** N+1 sigmas incl. terminal 0, f32-rounded values. */
  sigmas: Float64Array;
  /** N timesteps = sigma * 1000 (f32-rounded). */
  timesteps: Float64Array;
}

export function flux2Schedule(imageSeqLen: number, numSteps: number): Flux2Schedule {
  const mu = computeEmpiricalMu(imageSeqLen, numSteps);
  const em32 = f32(Math.exp(mu)); // numpy casts the python scalar to f32
  const sigmas = new Float64Array(numSteps + 1);
  const timesteps = new Float64Array(numSteps);
  const step = (1.0 / numSteps - 1.0) / (numSteps - 1); // np.linspace step, f64
  for (let i = 0; i < numSteps; i++) {
    // np.linspace(1, 1/N, N)[i] in f64 (start + i*step; last == stop exactly),
    // then .astype(np.float32)
    const lin = f32(i === numSteps - 1 ? 1.0 / numSteps : 1.0 + i * step);
    // exponential time shift, elementwise f32
    const inv = f32(f32(1.0 / lin) - 1.0);
    const s = f32(em32 / f32(em32 + inv));
    sigmas[i] = s;
    timesteps[i] = f32(s * 1000.0);
  }
  sigmas[numSteps] = 0.0;
  return { mu, sigmas, timesteps };
}

/** FlowMatchEulerDiscrete.step (non-stochastic): prev = sample + dt * v, f32.
 *  Writes into `sample` in place. */
export function eulerStep(
  sample: Float32Array,
  modelOutput: Float32Array,
  sigma: number,
  sigmaNext: number,
): void {
  const dt = f32(sigmaNext - sigma);
  for (let i = 0; i < sample.length; i++) {
    sample[i] = f32(sample[i] + f32(dt * modelOutput[i]));
  }
}
