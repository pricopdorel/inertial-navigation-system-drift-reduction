using Pkg
Pkg.activate(".")

# ============================================================================
# INFERENȚĂ ML1 CALIBRAT — încarcă modelul antrenat și evaluează pe trasee test
# Compară: mecanizare pură vs mecanizare calibrată vs ML1 (bias prezis).
# Salvează: bias prezis vs real (per axă), traiectorie, eroare poziție, RMSE.
# ============================================================================

include("SyntheticData.jl")
include("SciML_Models.jl")
include("Mechanization.jl")
include("Training_synth_ctx_calibrated_smooth.jl")

using .SyntheticData
using .SciML_Models
using .Mechanization
using .TrainingSynthCtxCalibrated
using LinearAlgebra
using Statistics
using Printf
using JLD2
using Rotations
using Plots

const MODEL_FILE = "synth_ctx_models_calib.jld2"
const WINDOW     = 30
const OUT_DIR    = "results_inference_ml1"
const FS         = 100.0

const TEST_SHAPES = [:circle, :figure8, :rectangle, :spiral, :random]
const TEST_DURATION = 100.0        # 100s pt vizualizare clară
const CORR_SEED = 777              # ≠ 100 (antrenare) → generalizare reală

println("="^70)
println("  INFERENȚĂ ML1 CALIBRAT — evaluare pe trasee de test")
println("="^70)

isfile(MODEL_FILE) || error("$MODEL_FILE lipsă. Rulează main_synth_calibrated.jl întâi.")
ML1 = JLD2.load(MODEL_FILE)["state"]
println("Model încărcat (window=$(ML1.window))")
println("  Calibrare acc:  $(ML1.bias_acc_calib)")
println("  Calibrare gyro: $(ML1.bias_gyro_calib)")

baf, bgf = make_bias_fns_ctx_calibrated(ML1)

function precompute_bias(r)
    W = ML1.window; n = r.n
    ba = zeros(n,3); bg = zeros(n,3)
    for i in 1:n
        lo = max(1, i-W+1)
        wa = r.acc[lo:i,:]; wg = r.gyro[lo:i,:]
        if size(wa,1) < W
            pad = W - size(wa,1)
            wa = vcat(repeat(wa[1:1,:],pad),wa); wg = vcat(repeat(wg[1:1,:],pad),wg)
        end
        ba[i,:] = baf(wa); bg[i,:] = bgf(wg)
    end
    return ba, bg
end

function mechanize(r, ba, bg)
    n = r.n; g_world = [0.0,0.0,-9.81]
    pos = zeros(n,3); vel = zeros(n,3)
    q = QuatRotation(r.quat[1,1],r.quat[1,2],r.quat[1,3],r.quat[1,4])
    pos[1,:] = r.pos[1,:]; vel[1,:] = r.vel[1,:]
    for i in 2:n
        dt = r.t[i]-r.t[i-1]; dt = dt<=0 ? 1/FS : dt
        gyro_i = r.gyro[i,:] .- bg[i,:]
        acc_i  = r.acc[i,:]  .- ba[i,:]
        dθ = gyro_i .* dt
        q = q * QuatRotation(RotXYZ(dθ[1],dθ[2],dθ[3]))
        acc_nav = q*acc_i .+ g_world
        vel[i,:] = vel[i-1,:] .+ acc_nav.*dt
        pos[i,:] = pos[i-1,:] .+ vel[i-1,:].*dt .+ 0.5.*acc_nav.*dt^2
    end
    return pos
end

pos_rmse(est, gt) = sqrt(mean(sum((est .- gt).^2, dims=2)))

summary_rows = String[]

for shape in TEST_SHAPES
    println("\n", "─"^60)
    println("Traseu: $shape")
    outdir = joinpath(OUT_DIR, string(shape)); mkpath(outdir)

    spec = TrajectorySpec(shape=shape, duration=TEST_DURATION/4, scale=150, laps=4, seed=999)
    corr = CorruptionSpec(bias_acc=BIAS_ACC_CALIB, bias_gyro=BIAS_GYRO_CALIB,
                          noise_acc=0.03, noise_gyro=0.0015,
                          rw_acc=0.005, rw_gyro=0.0003, seed=CORR_SEED)
    r = SyntheticData.generate_dataset([spec], corr; verbose=false)[1]

    ba0 = zeros(r.n,3); bg0 = zeros(r.n,3)                       # pură
    baC = repeat(ML1.bias_acc_calib', r.n); bgC = repeat(ML1.bias_gyro_calib', r.n)  # calibrată
    baM, bgM = precompute_bias(r)                                # ML1

    pos_pure  = mechanize(r, ba0, bg0)
    pos_calib = mechanize(r, baC, bgC)
    pos_ml1   = mechanize(r, baM, bgM)

    rmse_pure  = pos_rmse(pos_pure,  r.pos)
    rmse_calib = pos_rmse(pos_calib, r.pos)
    rmse_ml1   = pos_rmse(pos_ml1,   r.pos)
    @printf("  RMSE pură=%.2f  calibrată=%.2f  ML1=%.2f m\n", rmse_pure, rmse_calib, rmse_ml1)
    impr = 100*(1 - rmse_ml1/rmse_pure)
    @printf("  Îmbunătățire ML1 vs pură: %.1f%%\n", impr)
    push!(summary_rows, @sprintf("%-10s | pură %8.2f | calib %8.2f | ML1 %8.2f | −%.1f%%",
                                 string(shape), rmse_pure, rmse_calib, rmse_ml1, impr))

    p_traj = plot(r.pos[:,1], r.pos[:,2], label="Referință", color=:black, lw=2,
                  title="Traiectorie — $shape", xlabel="X (m)", ylabel="Y (m)", legend=:best)
    plot!(p_traj, pos_pure[:,1],  pos_pure[:,2],  label="Mecanizare pură", color=:red,    lw=1.5, ls=:dot)
    plot!(p_traj, pos_calib[:,1], pos_calib[:,2], label="Calibrată",        color=:orange, lw=1.5, ls=:dash)
    plot!(p_traj, pos_ml1[:,1],   pos_ml1[:,2],   label="ML1",              color=:green,  lw=2)
    savefig(p_traj, joinpath(outdir, "traiectorie.png"))

    e_pure  = [norm(pos_pure[i,:]  .- r.pos[i,:]) for i in 1:r.n]
    e_calib = [norm(pos_calib[i,:] .- r.pos[i,:]) for i in 1:r.n]
    e_ml1   = [norm(pos_ml1[i,:]   .- r.pos[i,:]) for i in 1:r.n]
    p_err = plot(r.t, e_pure,  label="Mecanizare pură", color=:red,    lw=1.5, ls=:dot,
                 title="Eroare poziție — $shape", xlabel="Timp (s)", ylabel="Eroare (m)", legend=:topleft)
    plot!(p_err, r.t, e_calib, label="Calibrată", color=:orange, lw=1.5, ls=:dash)
    plot!(p_err, r.t, e_ml1,   label="ML1",       color=:green,  lw=2)
    savefig(p_err, joinpath(outdir, "eroare_pozitie.png"))

    axn = ["X","Y","Z"]
    for (sensor, real_b, pred_b, unit) in
            [("acc", r.bias_acc_det, baM, "m/s²"), ("gyro", r.bias_gyro_det, bgM, "rad/s")]
        plt = plot(layout=(3,1), size=(800,700))
        for ax in 1:3
            plot!(plt[ax], r.t, real_b[:,ax], label="real", color=:black, lw=2)
            plot!(plt[ax], r.t, pred_b[:,ax], label="ML1 prezis", color=:green, lw=1.5,
                  ylabel="bias $(axn[ax]) ($unit)")
            ax==1 && title!(plt[ax], "Bias $sensor prezis vs real — $shape")
            ax==3 && xlabel!(plt[ax], "Timp (s)")
        end
        savefig(plt, joinpath(outdir, "bias_$(sensor).png"))
    end

    println("  → salvat în $outdir/")
end

println("\n", "="^70)
println("SUMAR RMSE (poziție, m) — corupție test seed=$CORR_SEED (≠ antrenare)")
println("="^70)
for row in summary_rows; println("  ", row); end
println("="^70)

open(joinpath(OUT_DIR, "sumar_rmse.txt"), "w") do io
    println(io, "SUMAR RMSE poziție (m) — test seed=$CORR_SEED")
    for row in summary_rows; println(io, row); end
end
println("\nGata. Rezultate în $OUT_DIR/")
