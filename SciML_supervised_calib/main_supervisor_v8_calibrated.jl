using Pkg
Pkg.activate(".")

include("SyntheticData.jl")
include("SciML_Models.jl")
include("Mechanization.jl")
include("Training_synth_ctx_calibrated.jl")
include("Training_supervisor_v8_calibrated.jl")

using .SyntheticData
using .SciML_Models
using .Mechanization
using .TrainingSynthCtxCalibrated
using .TrainingSupervisorV8Calibrated
using LinearAlgebra
using Statistics
using Printf
using JLD2
using Random

const ML1_FILE = "synth_ctx_models_calib.jld2"
const ML2_FILE = "synth_ctx_models_supervisor_v8_calib.jld2"

println("="^70)
println("  TRAIN SUPERVISOR V8 CALIBRATED — slope-window + anti-divergență")
println("  EXACT 15 trasee ca main.jl + hinge agresiv (λ=50)")
println("="^70)

println("\n[1] Încarc ML1 CALIBRATED din $ML1_FILE ...")
isfile(ML1_FILE) || error("$ML1_FILE nu există. Rulează main_synth_calibrated.jl întâi.")
@load ML1_FILE state
ml1 = state
println("   ML1 încărcat (window=$(ml1.window))")
println("   Calibrare acc:  $(ml1.bias_acc_calib)")
println("   Calibrare gyro: $(ml1.bias_gyro_calib)")

println("\n[2] Generare date — EXACT 15 trasee ca main.jl...")
train_specs = TrajectorySpec[
    TrajectorySpec(shape=:circle,    duration=300, scale=150, laps=1, seed=1),
    TrajectorySpec(shape=:circle,    duration=300, scale=250, laps=1, seed=2),
    TrajectorySpec(shape=:figure8,   duration=300, scale=150, laps=1, seed=3),
    TrajectorySpec(shape=:figure8,   duration=300, scale=200, laps=1, seed=4),
    TrajectorySpec(shape=:rectangle, duration=300, scale=150, laps=1, seed=5),
    TrajectorySpec(shape=:rectangle, duration=300, scale=200, laps=1, seed=6),
    TrajectorySpec(shape=:spiral,    duration=300, scale=150, laps=1, seed=7),
    TrajectorySpec(shape=:spiral,    duration=300, scale=250, laps=1, seed=8),
    TrajectorySpec(shape=:random,    duration=300, scale=150, laps=1, seed=11),
    TrajectorySpec(shape=:random,    duration=300, scale=180, laps=1, seed=12),
    TrajectorySpec(shape=:random,    duration=300, scale=120, laps=1, seed=13),
    TrajectorySpec(shape=:random,    duration=300, scale=200, laps=1, seed=14),
    TrajectorySpec(shape=:random,    duration=300, scale=170, laps=1, seed=15),
    TrajectorySpec(shape=:random,    duration=300, scale=130, laps=1, seed=16),
    TrajectorySpec(shape=:random,    duration=300, scale=220, laps=1, seed=17),
]

corr = CorruptionSpec(
    bias_acc=BIAS_ACC_CALIB, bias_gyro=BIAS_GYRO_CALIB,
    noise_acc=0.03, noise_gyro=0.0015, rw_acc=0.005, rw_gyro=0.0003, seed=100)

train_runs = SyntheticData.generate_dataset(train_specs, corr; verbose=false)
println("   $(length(train_runs)) trasee generate (EXACT ca main.jl)")

println("\n[3] Antrenare ML2 V8 CALIBRATED + SLOPE-WINDOW + HINGE AGRESIV...")
println("   slope_window=50, hinge λ=50 (anti-divergență tare)")
println("   Pornire integrare: ZERO (= calibrare în spațiu total)")

@time supervisor_v8 = TrainingSupervisorV8Calibrated.train_supervisor_v8_calib(
    train_runs, ml1;
    epochs=20, lr=3e-4,
    hidden=64, stride=8,
    window_ml2=10,
    dropout=0.3,
    dt=0.01,
    sigma_rw_acc=0.005,
    sigma_rw_gyro=0.0003,
    slope_window=50,
    checkpoint_dir="checkpoints_v8_calib",
    checkpoint_every=5)

@save ML2_FILE supervisor_v8
println("\n   ML2 V8 CALIB salvat → $ML2_FILE (cu drift=0 deocamdată)")
println("\n" * "="^70)
println("✓ Gata! Acum:")
println("   1. include(\"make_state_from_checkpoint_v8_calibrated.jl\") — va calibra drift-ul")
println("   2. include(\"inference_supervisor_v8_calibrated.jl\")")
println("="^70)
