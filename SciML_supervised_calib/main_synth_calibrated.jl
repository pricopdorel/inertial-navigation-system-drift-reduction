using Pkg
Pkg.activate(".")

include("SyntheticData.jl")
include("SciML_Models.jl")
include("Mechanization.jl")
include("Training_synth_ctx_calibrated.jl")

using .SyntheticData
using .SciML_Models
using .Mechanization
using .TrainingSynthCtxCalibrated
using LinearAlgebra
using Statistics
using Printf
using JLD2


const DATA_DIR  = "synthetic_data_calib"
const WINDOW    = 30
const MODEL_OUT = "synth_ctx_models_calib.jld2"
isdir(DATA_DIR) || mkpath(DATA_DIR)

println("="^70)
println("  TRAIN ML1 CALIBRATED — bias offset removal + 15 trasee")
println("="^70)
println("  Bias calibrare acc:  $(BIAS_ACC_CALIB)")
println("  Bias calibrare gyro: $(BIAS_GYRO_CALIB)")
println("="^70)

println("\n[1] Generare date — 8 forme + 7 random...")
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

train_runs = SyntheticData.generate_dataset(train_specs, corr; verbose=true)
println("   $(length(train_runs)) trasee generate")

println("\n[2] Antrenare ML1 CALIBRATED (LSTM, 25 epoci)...")
@time state = train_bias_ctx_calibrated(train_runs;
                    window=WINDOW,
                    epochs=25,
                    lr=1e-3,
                    hidden=128,
                    stride=5)

@save MODEL_OUT state
println("\n   ML1 CALIBRATED salvat → $MODEL_OUT")
println("\n" * "="^70)
println("✓ Gata! 8 forme + 7 random.")
println("  Următorul pas: antrenează ML2 V8 → include(\"main_supervisor_v8_calibrated.jl\")")
println("="^70)