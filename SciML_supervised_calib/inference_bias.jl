using Pkg
Pkg.activate(".")

include("SyntheticData.jl")
include("SciML_Models.jl")
include("Mechanization.jl")
include("Training_synth_ctx_calibrated.jl")
include("Training_supervisor_v8_calibrated.jl")
include("precompute_ml1_ratelimited.jl")


using .SyntheticData
using .SciML_Models
using .Mechanization
using .TrainingSynthCtxCalibrated
using .TrainingSupervisorV8Calibrated
using LinearAlgebra
using Statistics
using Printf
using Plots
using JLD2
using Random
using Rotations

const ML1_FILE = "synth_ctx_models_calib.jld2"
const ML2_FILE = "synth_ctx_models_supervisor_v8_calib.jld2"
const RES_DIR  = "results_supervisor_v8_calib"
isdir(RES_DIR) || mkpath(RES_DIR)

function shape_dirs(shape)
    base = joinpath(RES_DIR, String(shape))
    plotd = joinpath(base, "plot")
    errd  = joinpath(base, "error")
    biasd = joinpath(base, "bias")
    for d in (base, plotd, errd, biasd); isdir(d) || mkpath(d); end
    return plotd, errd, biasd
end

const TEST_SHAPES = [:circle, :figure8, :rectangle, :spiral, :random]
const TEST_DURATION = 25
const TEST_LAPS = 4
const CORR = CorruptionSpec(
    bias_acc=BIAS_ACC_CALIB, bias_gyro=BIAS_GYRO_CALIB,
    noise_acc=0.03, noise_gyro=0.0015, rw_acc=0.005, rw_gyro=0.0003, seed=0)

traj_rmse(pred,gt) = (n=min(size(pred,1),size(gt,1));
                     sqrt(mean(sum((pred[1:n,:].-gt[1:n,:]).^2,dims=2))))
traj_length(pos) = sum(norm(pos[i,:].-pos[i-1,:]) for i in 2:size(pos,1))

function precompute_ml1_calib_bias(r, ml1)
    baf, bgf = make_bias_fns_ctx_calibrated(ml1)
    W = ml1.window; n = r.n
    ba = zeros(n,3); bg = zeros(n,3)
    for i in 1:n
        lo = max(1, i-W+1)
        wa = r.acc[lo:i,:]; wg = r.gyro[lo:i,:]
        if size(wa,1) < W
            pad = W - size(wa,1)
            wa = vcat(repeat(wa[1:1,:],pad),wa); wg = vcat(repeat(wg[1:1,:],pad),wg)
        end
        ba[i,:]=baf(wa); bg[i,:]=bgf(wg)
    end
    return ba, bg
end

function run_mechanization_pure(r, ba_pred, bg_pred)
    n = r.n
    g_world = [0.0, 0.0, -9.81]
    est_pos = zeros(n, 3); est_vel = zeros(n, 3)
    q = QuatRotation(r.quat[1,1], r.quat[1,2], r.quat[1,3], r.quat[1,4])
    est_pos[1,:] = r.pos[1,:]; est_vel[1,:] = r.vel[1,:]
    for i in 2:n
        dt = r.t[i] - r.t[i-1]; dt = dt<=0 ? 0.01 : dt
        gyro_i = r.gyro[i,:] .- bg_pred[i,:]
        acc_i  = r.acc[i,:]  .- ba_pred[i,:]
        dθ = gyro_i .* dt
        q = q * QuatRotation(RotXYZ(dθ[1], dθ[2], dθ[3]))
        acc_nav = q * acc_i .+ g_world
        est_vel[i,:] = est_vel[i-1,:] .+ acc_nav .* dt
        est_pos[i,:] = est_pos[i-1,:] .+ est_vel[i-1,:] .* dt .+ 0.5 .* acc_nav .* dt^2
    end
    return est_pos, est_vel
end

function validate_shape(shape, ml1, sup_v8, rng)
    plot_dir, err_dir, bias_dir = shape_dirs(shape)

    sd_traj = 3
    sd_corr = 103
    
    if shape == :spiral
        shname = "Spirală"   
    end 
    if shape == :circle
        shname = "Cerc"
    end    
    if shape == :figure8
        shname = "Fundă"
    end
    if shape == :rectangle
        shname = "Dreptunghi"
    end
    if shape == :random
        shname = "Random"
    end

    scale = shape == :spiral ? 180.0 : (120.0 + 80*rand(rng))
    spec = TrajectorySpec(shape=shape, duration=TEST_DURATION, scale=scale,
                          laps=TEST_LAPS, seed=sd_traj)
    corr = CorruptionSpec(
        bias_acc=CORR.bias_acc, bias_gyro=CORR.bias_gyro,
        noise_acc=CORR.noise_acc, noise_gyro=CORR.noise_gyro,
        rw_acc=CORR.rw_acc, rw_gyro=CORR.rw_gyro, seed=sd_corr)

    r = SyntheticData.generate_dataset([spec], corr; verbose=false)[1]

    zba = zeros(r.n, 3); zbg = zeros(r.n, 3)
    epos0, _ = run_mechanization_pure(r, zba, zbg)

    ba1, bg1 = precompute_ml1_calib_bias_limited(r, ml1;
                rw_acc=0.005, rw_gyro=0.0003, k_tol=4.0)
    epos1, _ = run_mechanization_pure(r, ba1, bg1)

    ba3, bg3 = integrate_bias_v8_calib(r, sup_v8; residual_scale=1.0)
    epos3, _ = run_mechanization_pure(r, ba3, bg3)

    L = traj_length(r.pos)
    rmse0 = traj_rmse(epos0, r.pos)
    rmse1 = traj_rmse(epos1, r.pos)
    rmse3 = traj_rmse(epos3, r.pos)

    p = plot(r.pos[:,1], r.pos[:,2], label="GT", lw=3, color=:black,
             xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal, legend=:best,
             title="$shname - Mec=$(round(rmse0,digits=1)) | ML bias=$(round(rmse1,digits=1)) | ML sup=$(round(rmse3,digits=1))")
    plot!(p, epos0[:,1], epos0[:,2], label="mecanizare pură", lw=2, color=:red, ls=:dot)
    plot!(p, epos1[:,1], epos1[:,2], label="ML bias", lw=2, color=:orange, ls=:dash)
    plot!(p, epos3[:,1], epos3[:,2], label="ML bias+sup", lw=2, color=:green)
    savefig(p, joinpath(plot_dir, "traj_$(shape).png"))

    nmin = min(size(epos0,1), size(epos1,1), size(epos3,1), r.n)
    err0 = [norm(epos0[i,:].-r.pos[i,:]) for i in 1:nmin]
    err1 = [norm(epos1[i,:].-r.pos[i,:]) for i in 1:nmin]
    err3 = [norm(epos3[i,:].-r.pos[i,:]) for i in 1:nmin]
    pe = plot(r.t[1:nmin], err0, label="mecanizare pură", lw=2, color=:red, ls=:dot,
              xlabel="Timp (s)", ylabel="Eroare (m)",
              title="$shname - Evoluția derivei", legend=:topleft)
    plot!(pe, r.t[1:nmin], err1, label="ML bias", lw=2, color=:orange, ls=:dash)
    plot!(pe, r.t[1:nmin], err3, label="ML bias+sup", lw=2, color=:green)
    savefig(pe, joinpath(err_dir, "error_$(shape).png"))

    axes_names = ["X", "Y", "Z"]

    for ax in 1:3
        pba = plot(r.t, r.bias_acc_det[:,ax], label="real", lw=2, color=:black,
                   xlabel="Timp (s)", ylabel="Bias acc $(axes_names[ax]) (m/s²)",
                   title="Bias acc $(axes_names[ax]) — $shape (calibrat)", legend=:best)
        plot!(pba, r.t, ba1[:,ax], label="ML bias", lw=1.5, color=:orange, ls=:dash)
        plot!(pba, r.t, ba3[:,ax], label="ML bias+sup", lw=2, color=:green)
        savefig(pba, joinpath(bias_dir, "bias_acc_$(axes_names[ax])_$(shape).png"))
    end

    for ax in 1:3
        pbg = plot(r.t, r.bias_gyro_det[:,ax], label="real", lw=2, color=:black,
                   xlabel="Timp (s)", ylabel="Bias gyro $(axes_names[ax]) (rad/s)",
                   title="Bias gyro $(axes_names[ax]) — $shape (calibrat)", legend=:best)
        plot!(pbg, r.t, bg1[:,ax], label="ML bias", lw=1.5, color=:orange, ls=:dash)
        plot!(pbg, r.t, bg3[:,ax], label="ML bias+sup", lw=2, color=:green)
        savefig(pbg, joinpath(bias_dir, "bias_gyro_$(axes_names[ax])_$(shape).png"))
    end

    return (shape=shape, L=L, rmse0=rmse0, rmse1=rmse1, rmse3=rmse3)
end

println("="^70)
println("  INFERENCE V8 CALIBRATED — GT vs ML1_calib vs ML1+V8_calib")
println("="^70)

isfile(ML1_FILE) || error("$ML1_FILE nu există.")
isfile(ML2_FILE) || error("$ML2_FILE nu există.")

@load ML1_FILE state
ml1 = state
@load ML2_FILE supervisor_v8
sup_v8 = supervisor_v8
println("   ML1 încărcat (window=$(ml1.window))")
println("   ML2 V8 încărcat (window_ml2=$(sup_v8.window_ml2))")
println("   Calibrare acc:  $(ml1.bias_acc_calib)")
println("   Calibrare gyro: $(ml1.bias_gyro_calib)")

rng = MersenneTwister(42)   

results = []
for shape in TEST_SHAPES
    println("\n[*] Validez forma: $shape ...")
    push!(results, validate_shape(shape, ml1, sup_v8, rng))
end

println("\n" * "="^78)
println("  TABEL — Mecanizare pură vs ML bias")
println("="^78)
@printf("  %-12s %-9s %-11s %-13s %-13s\n", "Formă", "Lung(m)", "Mec(m)", "ML1_calib(m)", "ML1+V8_calib(m)")
println("  " * "-"^66)
for x in results
    @printf("  %-12s %-9.0f %-11.2f %-13.2f %-13.2f\n", x.shape, x.L, x.rmse0, x.rmse1, x.rmse3)
end
println("="^78)

n3 = count(x -> x.rmse3 < x.rmse1, results)
d3 = mean(100*(x.rmse1 - x.rmse3)/max(x.rmse1,1e-9) for x in results)

red1 = mean(100*(x.rmse0 - x.rmse1)/max(x.rmse0,1e-9) for x in results)
red3 = mean(100*(x.rmse0 - x.rmse3)/max(x.rmse0,1e-9) for x in results)

@printf("\n  V8 vs ML1: îmbunătățire medie %+.2f%%, %d/%d forme mai bune\n",
        d3, n3, length(results))
@printf("  Reducere drift vs mecanizare pură: ML1 %+.1f%%, ML1+V8 %+.1f%%\n",
        red1, red3)

println("\n✓ Inference V8 CALIB complet! Grafice în $RES_DIR/ (structurate pe formă)")
for shape in TEST_SHAPES
    base = joinpath(RES_DIR, String(shape))
    println("   - $base/")
    println("       plot/  → traiectorie 2D XY")
    println("       error/ → eroare poziție în timp")
    println("       bias/  → bias acc + gyro pe 3 axe (X, Y, Z)")
end