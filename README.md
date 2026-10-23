# IMU Bias and Drift Reduction with Machine Learning (Julia / Lux)

Bachelor's thesis project. The goal is to reduce the bias and drift of a low-cost MEMS IMU (accelerometer + gyroscope) using neural networks, and to fuse the corrected signal in an adaptive Extended Kalman Filter (EKF). The work follows the approach of McManus (MIT, 2024) and is implemented entirely in Julia with [Lux.jl](https://github.com/LuxDL/Lux.jl).

> **Status:** research code. Results are honest rather than spectacular: the first network (ML1) works as a bias estimator, the second network (ML2) improves the estimate locally but is fundamentally limited globally (see [Results and limitations](#results-and-limitations)).

---

## Table of contents

1. [Overview](#overview)
2. [Repository structure](#repository-structure)
3. [Synthetic data and sensor model](#synthetic-data-and-sensor-model)
4. [Models](#models)
5. [Adaptive EKF](#adaptive-ekf)
6. [Installation](#installation)
7. [How to run](#how-to-run)
8. [Outputs](#outputs)
9. [Results and limitations](#results-and-limitations)
10. [Known issues](#known-issues)
11. [References](#references)

---

## Overview

The project contains three pipelines. The main one is Pipeline 3.

| Pipeline | Data | Description | Status |
|---|---|---|---|
| 1 | Real (OxIOD) | SciML approach following McManus | Done, negative result |
| 2 | Real (OxIOD) | Deep 6-DOF inertial odometry (Lima-style) | Done |
| **3** | **Synthetic** | **ML bias estimation + adaptive EKF** | **Main focus** |

### Pipeline 3 at a glance

```
raw IMU ──► static calibration ──► ML1 (LSTM) ──► ML2 (BiLSTM, slope) ──► mechanization ──► position / attitude
                                                                                │
                                                       adaptive EKF + sparse GPS reference
```

1. **Static calibration** removes the known constant offset of the sensor.
2. **ML1** (LSTM) estimates the remaining slowly varying bias (random walk) from a 300 ms window.
3. **ML2** (Bi-LSTM supervisor) predicts the *slope* of the bias over 0.5 s blocks and integrates it, anchored to ML1 by a leak term.
4. **Mechanization** integrates the corrected IMU to obtain orientation (quaternion), velocity and position.
5. **Adaptive EKF** (6-state) corrects drift using an external reference, requested only when position uncertainty grows too large.

---

## Repository structure

```
SciML_supervised_calib/
├── SyntheticData.jl                          # trajectory generator + IMU corruption model
├── SciML_Models.jl                           # LSTM / BiLSTM architectures (Lux)
├── Mechanization.jl                          # strapdown mechanization (quaternion)
├── EKF_adaptive.jl                           # adaptive EKF with covariance-based acquisition
│
├── Training_synth_ctx_calibrated.jl          # ML1 module (calibrated)
├── main_synth_calibrated.jl                  # ML1 training script
│
├── Training_supervisor_v8_calibrated.jl      # ML2 (V8) module: slope window, NLL+hinge loss, leak
├── main_supervisor_v8_calibrated.jl          # ML2 training script
│
├── inference_supervisor_v8_calibrated.jl     # inference: trajectory, error, bias on X/Y/Z
├── inference_bias.jl                         # inference: bias per axis (acc + gyro), CSV export
├── inference_odometric.jl                    # inference: position X/Y/Z + roll/pitch/yaw, CSV export
│
├── simulator_3d.jl                           # real-time 3D simulator (GLMakie)
└── (other scripts: apel_mecanizare.jl, apel_zaru_fig8.jl, etc.)
```

---

## Synthetic data and sensor model

Ground-truth trajectories are generated analytically (circle, figure-eight, rectangle, spiral, smooth random), with orientation as unit quaternions. An *inverse mechanization* step computes the ideal IMU signal that would produce that trajectory, and the signal is then corrupted with three error sources:

- **Offset** (constant bias)
- **Random walk** (slow bias drift)
- **White noise**

Corruption parameters (100 Hz sampling, `dt = 0.01 s`):

| Error | Accelerometer | Gyroscope |
|---|---|---|
| Offset (X, Y, Z) | `[0.10, -0.08, 0.06]` m/s² | `[0.010, -0.008, 0.006]` rad/s |
| White noise (std) | 0.03 m/s² | 0.0015 rad/s |
| Random walk | 0.005 m/s²/√s | 0.0003 rad/s/√s |

The values were chosen to be in the order of magnitude of consumer MEMS IMUs such as the InvenSense MPU-6050 (see Rudyk et al., 2020, who characterize the MPU-6050 with Allan variance). If you cite exact datasheet or paper figures in your own work, verify them against the original sources.

Training set: 15 trajectories of 300 s (2 circles, 2 figure-eights, 2 rectangles, 2 spirals, 7 random), corruption seed 100.

---

## Models

### ML1: bias estimator

- Architecture: `LSTM(3 → 128)` followed by `Dense(128 → 128, tanh)` and `Dense(128 → 3)`. One model for the accelerometer, one for the gyroscope (different seeds).
- Input: a window of 30 calibrated samples (300 ms), z-score normalized per channel.
- Target: the deterministic bias minus the calibration offset (the random-walk part only).
- Loss: MSE. Optimizer: Adam with step decay. Batch size 256, window stride 5, 25 epochs.
- The reference bias is used **only inside the loss**, never as network input.
- Output: residual bias; the total bias is `calibration offset + residual`.

### ML2 (V8): slope supervisor

- Architecture: Bi-LSTM (6 → 64, dropout 0.3) with a `tanh` output scaled to ±3σ in physical units.
- Predicts the **average slope** of the bias over a 50-sample (0.5 s) block, applied as a constant over that block (block-wise integration).
- Loss: Gaussian NLL with the known random-walk σ, plus a hinge penalty outside ±3σ.
- Integration starts from zero residual (the calibration offset), applies a systematic-drift correction calibrated on separate trajectories, and uses a **leak toward ML1** (`α = 0.001`, time constant ≈ 10 s) to reduce long-term divergence.

---

## Adaptive EKF

State `x = [position(3); velocity(3)]` (6D). Orientation is propagated separately as a quaternion.

- **Prediction:** second-order kinematic propagation, covariance `P ← F P Fᵀ + Q`.
- **Update (with reference):** classic Kalman update with position (and optionally velocity) observations.
- **Adaptive acquisition:** a reference measurement is requested when `tr(P_pos) > p_threshold` (default 25 m²) or after `period_max` steps (safety net).
- **Without reference:** behaves like dead reckoning with covariance propagation.

---

## Installation

Requirements: Julia 1.10+ (developed on 1.12).

```julia
julia> using Pkg
julia> Pkg.activate(".")
julia> Pkg.add(["Lux", "Zygote", "Optimisers", "ComponentArrays", "JLD2",
                "Rotations", "Plots", "Statistics", "Printf", "Random"])
# For the 3D simulator:
julia> Pkg.add("GLMakie")
```

---

## How to run

> **Run from the Julia REPL** (`julia`, then `include(...)`). Do **not** use the VS Code debugger: it evaluates top-level code differently and can fail on some macros.

```julia
# 1. Train ML1 (calibrated), ~30-60 min
include("main_synth_calibrated.jl")

# 2. Train ML2 (V8), ~1-2 h
include("main_supervisor_v8_calibrated.jl")

# 3. Convert checkpoint and calibrate systematic drift
include("make_state_from_checkpoint_v8_calibrated.jl")

# 4. Inference
include("inference_supervisor_v8_calibrated.jl")   # trajectory, error, bias on X/Y/Z
include("inference_bias.jl")                        # bias per axis + CSV
include("inference_odometric.jl")                   # position + attitude + CSV
```

Both training scripts support checkpoint/resume. Delete `checkpoints_v8_calib/` before retraining ML2 from scratch.

---

## Outputs

```
results_supervisor_v8_calib/   plot/  error/  bias/        (bias: acc and gyro, X/Y/Z)
results_bias/                  bias/  csv/
results_odometric/             plot/  error/  position/  attitude/  csv/
```

Each comparison plot shows ground truth, ML1 alone, and ML1 + V8. CSV files contain time series for real, ML1 and V8 signals, so they can be analyzed outside Julia.

---

## Results and limitations

Example run, RMSE of position in meters (trajectory lengths of several kilometers; results vary with random seeds):

| Shape | Length (m) | ML1 calibrated | ML1 + V8 |
|---|---|---|---|
| circle | 6462 | 256.17 | 60.54 |
| figure8 | 7387 | 543.28 | 755.71 |
| rectangle | 5626 | 176.66 | 106.26 |
| spiral | 7399 | 328.28 | 303.52 |
| random | 270 | 1155.83 | 1194.43 |

What the experiments show:

- ML1 tracks the bias level but oscillates with the trajectory shape, a sign of partial overfitting to the 15 training trajectories.
- ML2 corrects local oscillations and short-term slopes well, but its integrated estimate can diverge globally. The leak toward ML1 and the drift correction reduce this but do not remove it.
- The ML2 loss stagnates close to the value of a constant predictor. This is consistent with the theoretical limit: the slope of a random walk over a short interval is essentially unpredictable from noisy observations, so the supervisor cannot extract much more than ML1 already provides.
- Results are mixed across shapes (V8 is worse on some). They should be read as a study of what is and is not learnable, not as a solved problem.

Suggestions for future work: more training trajectories with a smaller network (the window-to-parameter ratio is currently low), validation on real sensor data, and a probabilistic output head.

---

## Known issues

- File name capitalization (`SyntheticData.jl` vs `Syntheticdata.jl`) matters on Linux/macOS. Use one spelling consistently in the `include` calls.
- `@printf` in Julia 1.12 requires a literal format string; use `@sprintf` when the format is built at runtime.
- Julia code cannot be verified here beyond bracket balance in some environments; run each script once from the REPL and report errors.

---

## References

- McManus, 2024. MIT thesis on SciML-based IMU error correction (primary reference).
- Rudyk et al., 2020. *Strapdown Inertial Navigation Systems for Positioning Mobile Robots: MEMS Gyroscopes Random Errors Analysis Using Allan Variance Method*, Sensors 20(17):4841.
- Bistrovs and Kluga, 2012. Adaptive Kalman filtering for navigation.
- Kalman, 1960. *A New Approach to Linear Filtering and Prediction Problems*.
- Kendall and Gal, 2017. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*
- Sola, 2017. *Quaternion kinematics for the error-state Kalman filter*, arXiv:1711.02508.
- Groves, 2013. *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*.
- Titterton and Weston, 2004. *Strapdown Inertial Navigation Technology*.

