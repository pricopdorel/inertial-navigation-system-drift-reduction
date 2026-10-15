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
using Plots
using JLD2
using Random
using Rotations

const ML1_FILE = "synth_ctx_models_calib.jld2"
const ML2_FILE = "synth_ctx_models_supervisor_v8_calib.jld2"
const RES_DIR  = "results_odometric"
const PLOT_DIR = joinpath(RES_DIR, "plot")        # traiectorie 2D XY
const ERR_DIR  = joinpath(RES_DIR, "error")       # eroare pozitie totala
const POS_DIR  = joinpath(RES_DIR, "position")    # X, Y, Z separat
const ATT_DIR  = joinpath(RES_DIR, "attitude")    # roll, pitch, yaw
const CSV_DIR  = joinpath(RES_DIR, "csv")
for d in (RES_DIR, PLOT_DIR, ERR_DIR, POS_DIR, ATT_DIR, CSV_DIR)
    isdir(d) || mkpath(d)
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

function quat_to_rpy(qw, qx, qy, qz)
    q = QuatRotation(qw, qx, qy, qz)
    r = RotXYZ(q)
    return r.theta1, r.theta2, r.theta3
end

function quat_array_to_rpy(quat_n4::AbstractMatrix)
    n = size(quat_n4, 1)
    rpy = zeros(n, 3)
    for i in 1:n
        roll, pitch, yaw = quat_to_rpy(quat_n4[i,1], quat_n4[i,2],
                                       quat_n4[i,3], quat_n4[i,4])
        rpy[i,:] = [roll, pitch, yaw]
    end
    return rpy
end

# Unwrap unghiuri (elimină salturile ±2π)
function unwrap_angles(angles::AbstractVector)
    n = length(angles)
    out = copy(angles)
    for i in 2:n
        while out[i] - out[i-1] > π
            out[i] -= 2π
        end
        while out[i] - out[i-1] < -π
            out[i] += 2π
        end
    end
    return out
end

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

# ============================================================================
# Mecanizare cu returnare quaternioni (pentru extragere RPY)
# ============================================================================
function run_mechanization_with_quat(r, ba_pred, bg_pred)
    n = r.n
    g_world = [0.0, 0.0, -9.81]
    est_pos = zeros(n, 3); est_vel = zeros(n, 3); est_quat = zeros(n, 4)
    q = QuatRotation(r.quat[1,1], r.quat[1,2], r.quat[1,3], r.quat[1,4])
    est_pos[1,:] = r.pos[1,:]; est_vel[1,:] = r.vel[1,:]
    est_quat[1,:] = [q.q.s, q.q.v1, q.q.v2, q.q.v3]
    for i in 2:n
        dt = r.t[i] - r.t[i-1]; dt = dt<=0 ? 0.01 : dt
        gyro_i = r.gyro[i,:] .- bg_pred[i,:]
        acc_i  = r.acc[i,:]  .- ba_pred[i,:]
        dθ = gyro_i .* dt
        q = q * QuatRotation(RotXYZ(dθ[1], dθ[2], dθ[3]))
        acc_nav = q * acc_i .+ g_world
        est_vel[i,:] = est_vel[i-1,:] .+ acc_nav .* dt
        est_pos[i,:] = est_pos[i-1,:] .+ est_vel[i-1,:] .* dt .+ 0.5 .* acc_nav .* dt^2
        est_quat[i,:] = [q.q.s, q.q.v1, q.q.v2, q.q.v3]
    end
    return est_pos, est_vel, est_quat
end

function save_odometric_csv(shape, r, epos1, epos3, rpy_real, rpy1, rpy3)
    nmin = min(size(epos1,1), size(epos3,1), r.n)
    path = joinpath(CSV_DIR, "odometric_$(shape).csv")
    open(path, "w") do io
        println(io, "time,pos_x_real,pos_y_real,pos_z_real,pos_x_ml1,pos_y_ml1,pos_z_ml1,pos_x_v8,pos_y_v8,pos_z_v8,roll_real,pitch_real,yaw_real,roll_ml1,pitch_ml1,yaw_ml1,roll_v8,pitch_v8,yaw_v8")
        for i in 1:nmin
            line = @sprintf("%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                r.t[i],
                r.pos[i,1], r.pos[i,2], r.pos[i,3],
                epos1[i,1], epos1[i,2], epos1[i,3],
                epos3[i,1], epos3[i,2], epos3[i,3],
                rpy_real[i,1], rpy_real[i,2], rpy_real[i,3],
                rpy1[i,1], rpy1[i,2], rpy1[i,3],
                rpy3[i,1], rpy3[i,2], rpy3[i,3])
            println(io, line)
        end
    end
    return path
end

function validate_shape_odom(shape, ml1, sup_v8, rng)
    sd_traj = rand(rng, 1:1_000_000)
    sd_corr = rand(rng, 1:1_000_000)
    scale = shape == :spiral ? 180.0 : (120.0 + 80*rand(rng))
    spec = TrajectorySpec(shape=shape, duration=TEST_DURATION, scale=scale,
                          laps=TEST_LAPS, seed=sd_traj)
    corr = CorruptionSpec(
        bias_acc=CORR.bias_acc, bias_gyro=CORR.bias_gyro,
        noise_acc=CORR.noise_acc, noise_gyro=CORR.noise_gyro,
        rw_acc=CORR.rw_acc, rw_gyro=CORR.rw_gyro, seed=sd_corr)

    r = SyntheticData.generate_dataset([spec], corr; verbose=false)[1]

    ba1, bg1 = precompute_ml1_calib_bias(r, ml1)
    epos1, _, equat1 = run_mechanization_with_quat(r, ba1, bg1)

    ba3, bg3 = integrate_bias_v8_calib(r, sup_v8; residual_scale=1.0)
    epos3, _, equat3 = run_mechanization_with_quat(r, ba3, bg3)

    L = traj_length(r.pos)
    rmse1 = traj_rmse(epos1, r.pos)
    rmse3 = traj_rmse(epos3, r.pos)

    rpy_real = quat_array_to_rpy(r.quat)
    rpy1     = quat_array_to_rpy(equat1)
    rpy3     = quat_array_to_rpy(equat3)

    rpy_real_unw = copy(rpy_real)
    rpy1_unw     = copy(rpy1)
    rpy3_unw     = copy(rpy3)
    rpy_real_unw[:,3] = unwrap_angles(rpy_real[:,3])
    rpy1_unw[:,3]     = unwrap_angles(rpy1[:,3])
    rpy3_unw[:,3]     = unwrap_angles(rpy3[:,3])

    nmin = min(size(rpy1,1), size(rpy3,1), size(rpy_real,1), r.n)
    rmse_roll1  = sqrt(mean((rpy1[1:nmin,1] .- rpy_real[1:nmin,1]).^2))
    rmse_roll3  = sqrt(mean((rpy3[1:nmin,1] .- rpy_real[1:nmin,1]).^2))
    rmse_pitch1 = sqrt(mean((rpy1[1:nmin,2] .- rpy_real[1:nmin,2]).^2))
    rmse_pitch3 = sqrt(mean((rpy3[1:nmin,2] .- rpy_real[1:nmin,2]).^2))
    rmse_yaw1   = sqrt(mean((rpy1_unw[1:nmin,3] .- rpy_real_unw[1:nmin,3]).^2))
    rmse_yaw3   = sqrt(mean((rpy3_unw[1:nmin,3] .- rpy_real_unw[1:nmin,3]).^2))

    p = plot(r.pos[:,1], r.pos[:,2], label="GT", lw=3, color=:black,
             xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal, legend=:best,
             title="$shape — ML1=$(round(rmse1,digits=1)) | V8=$(round(rmse3,digits=1))")
    plot!(p, epos1[:,1], epos1[:,2], label="ML1 calib", lw=2, color=:orange, ls=:dash)
    plot!(p, epos3[:,1], epos3[:,2], label="ML1+V8 calib", lw=2, color=:green)
    savefig(p, joinpath(PLOT_DIR, "traj_$(shape).png"))

    err1 = [norm(epos1[i,:].-r.pos[i,:]) for i in 1:nmin]
    err3 = [norm(epos3[i,:].-r.pos[i,:]) for i in 1:nmin]
    pe = plot(r.t[1:nmin], err1, label="ML1 calib", lw=2, color=:orange, ls=:dash,
              xlabel="Timp (s)", ylabel="Eroare (m)",
              title="Drift — $shape", legend=:topleft)
    plot!(pe, r.t[1:nmin], err3, label="ML1+V8 calib", lw=2, color=:green)
    savefig(pe, joinpath(ERR_DIR, "error_$(shape).png"))

    axes_xyz = ["X", "Y", "Z"]
    for ax in 1:3
        ppos = plot(r.t[1:nmin], r.pos[1:nmin,ax], label="$(axes_xyz[ax]) real", lw=2, color=:black,
                    xlabel="Timp (s)", ylabel="$(axes_xyz[ax]) (m)",
                    title="Poziție $(axes_xyz[ax]) — $shape", legend=:best)
        plot!(ppos, r.t[1:nmin], epos1[1:nmin,ax], label="$(axes_xyz[ax]) ML1 calib",
              lw=1.5, color=:orange, ls=:dash)
        plot!(ppos, r.t[1:nmin], epos3[1:nmin,ax], label="$(axes_xyz[ax]) ML1+V8 calib",
              lw=2, color=:green)
        savefig(ppos, joinpath(POS_DIR, "pos_$(axes_xyz[ax])_$(shape).png"))
    end

    angle_names = ["roll", "pitch"]
    for (idx, name) in enumerate(angle_names)
        patt = plot(r.t[1:nmin], rad2deg.(rpy_real[1:nmin,idx]), label="$(name) real",
                    lw=2, color=:black,
                    xlabel="Timp (s)", ylabel="$(name) (°)",
                    title="$(name) — $shape", legend=:best)
        plot!(patt, r.t[1:nmin], rad2deg.(rpy1[1:nmin,idx]), label="$(name) ML1 calib",
              lw=1.5, color=:orange, ls=:dash)
        plot!(patt, r.t[1:nmin], rad2deg.(rpy3[1:nmin,idx]), label="$(name) ML1+V8 calib",
              lw=2, color=:green)
        savefig(patt, joinpath(ATT_DIR, "$(name)_$(shape).png"))
    end

    pyaw = plot(r.t[1:nmin], rad2deg.(rpy_real_unw[1:nmin,3]), label="yaw real", lw=2, color=:black,
                xlabel="Timp (s)", ylabel="yaw (°)",
                title="yaw — $shape", legend=:best)
    plot!(pyaw, r.t[1:nmin], rad2deg.(rpy1_unw[1:nmin,3]), label="yaw ML1 calib",
          lw=1.5, color=:orange, ls=:dash)
    plot!(pyaw, r.t[1:nmin], rad2deg.(rpy3_unw[1:nmin,3]), label="yaw ML1+V8 calib",
          lw=2, color=:green)
    savefig(pyaw, joinpath(ATT_DIR, "yaw_$(shape).png"))

    csv_path = save_odometric_csv(shape, r, epos1, epos3, rpy_real, rpy1, rpy3)

    return (shape=shape, L=L,
            rmse_pos1=rmse1, rmse_pos3=rmse3,
            rmse_roll1=rad2deg(rmse_roll1), rmse_roll3=rad2deg(rmse_roll3),
            rmse_pitch1=rad2deg(rmse_pitch1), rmse_pitch3=rad2deg(rmse_pitch3),
            rmse_yaw1=rad2deg(rmse_yaw1), rmse_yaw3=rad2deg(rmse_yaw3),
            csv=csv_path)
end

println("="^70)
println("  INFERENCE ODOMETRIC — pos X/Y/Z + roll/pitch/yaw")
println("="^70)

isfile(ML1_FILE) || error("$ML1_FILE nu există.")
isfile(ML2_FILE) || error("$ML2_FILE nu există.")

@load ML1_FILE state
ml1 = state
@load ML2_FILE supervisor_v8
sup_v8 = supervisor_v8
println("   ML1 încărcat (window=$(ml1.window))")
println("   ML2 V8 încărcat (window_ml2=$(sup_v8.window_ml2))")

rng = MersenneTwister(round(Int, time()*1000) % 2_000_000_000)

results = []
for shape in TEST_SHAPES
    println("\n[*] Odometric: $shape ...")
    res = validate_shape_odom(shape, ml1, sup_v8, rng)
    println("   ✓ CSV: $(res.csv)")
    push!(results, res)
end

println("\n" * "="^70)
println("  TABEL — RMSE Poziție 3D (m)")
println("="^70)
@printf("  %-12s  %-9s  %-13s  %-13s\n", "Formă", "Lung(m)", "ML1 calib(m)", "V8 calib(m)")
println("  " * "-"^55)
for x in results
    @printf("  %-12s  %-9.0f  %-13.2f  %-13.2f\n", x.shape, x.L, x.rmse_pos1, x.rmse_pos3)
end
println("="^70)

println("\n" * "="^80)
println("  TABEL — RMSE Atitudine (grade) — Roll / Pitch / Yaw")
println("="^80)
@printf("  %-12s  %-30s  %-30s\n", "Formă", "ML1 (R / P / Y)", "V8 (R / P / Y)")
println("  " * "-"^78)
for x in results
    @printf("  %-12s  %.3f / %.3f / %.3f      %.3f / %.3f / %.3f\n",
        x.shape,
        x.rmse_roll1, x.rmse_pitch1, x.rmse_yaw1,
        x.rmse_roll3, x.rmse_pitch3, x.rmse_yaw3)
end
println("="^80)

n3 = count(x -> x.rmse_pos3 < x.rmse_pos1, results)
d3 = mean(100*(x.rmse_pos1 - x.rmse_pos3)/max(x.rmse_pos1,1e-9) for x in results)
@printf("\n  V8 calib: îmbunătățire medie poziție %+.2f%%, %d/%d forme mai bune\n",
        d3, n3, length(results))

println("\n✓ Inference ODOMETRIC complet în $RES_DIR/:")
println("   - $PLOT_DIR/      → traiectorie 2D XY")
println("   - $ERR_DIR/       → eroare poziție în timp")
println("   - $POS_DIR/       → poziție X, Y, Z separat")
println("   - $ATT_DIR/       → roll, pitch, yaw (grade)")
println("   - $CSV_DIR/       → CSV pos + rpy per formă")
