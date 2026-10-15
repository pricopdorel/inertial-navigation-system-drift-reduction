using Pkg
Pkg.activate(".")


include("SyntheticData.jl")
include("SciML_Models.jl")
include("Mechanization.jl")
include("EKF_classic.jl")
include("Training_synth_ctx_calibrated.jl")
include("Training_supervisor_v8_calibrated.jl")
include("precompute_ml1_ratelimited.jl")

using .SyntheticData
using .SciML_Models
using .Mechanization
using .EKFClassic
using .TrainingSynthCtxCalibrated
using .TrainingSupervisorV8Calibrated
using LinearAlgebra
using Statistics
using Printf
using JLD2
using Random
using Rotations

const ML1_FILE = "synth_ctx_models_calib.jld2"
const ML2_FILE = "synth_ctx_models_supervisor_v8_calib.jld2"

const TEST_SHAPES   = [:circle, :figure8, :rectangle, :spiral, :random]
const TEST_DURATION = 25
const TEST_LAPS     = 4
const CORR = CorruptionSpec(
    bias_acc=BIAS_ACC_CALIB, bias_gyro=BIAS_GYRO_CALIB,
    noise_acc=0.03, noise_gyro=0.0015, rw_acc=0.005, rw_gyro=0.0003, seed=0)

traj_rmse(pred,gt) = (n=min(size(pred,1),size(gt,1));
                     sqrt(mean(sum((pred[1:n,:].-gt[1:n,:]).^2,dims=2))))
traj_length(pos) = sum(norm(pos[i,:].-pos[i-1,:]) for i in 2:size(pos,1))

function ml1_bias(r, ml1)
    return precompute_ml1_calib_bias_limited(r, ml1; rw_acc=0.005, rw_gyro=0.0003, k_tol=4.0)
end

function mechanize_state(r, ba, bg)
    n = r.n; g_world = [0.0,0.0,-9.81]
    pos = zeros(n,3); vel = zeros(n,3); quat = zeros(n,4)
    q = QuatRotation(r.quat[1,1],r.quat[1,2],r.quat[1,3],r.quat[1,4])
    pos[1,:]=r.pos[1,:]; vel[1,:]=r.vel[1,:]; quat[1,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
    for i in 2:n
        dt = r.t[i]-r.t[i-1]; dt = dt<=0 ? 0.01 : dt
        gyro_i = r.gyro[i,:] .- bg[i,:]; acc_i = r.acc[i,:] .- ba[i,:]
        dθ = gyro_i .* dt; q = q*QuatRotation(RotXYZ(dθ[1],dθ[2],dθ[3]))
        acc_nav = q*acc_i .+ g_world
        vel[i,:] = vel[i-1,:] .+ acc_nav.*dt
        pos[i,:] = pos[i-1,:] .+ vel[i-1,:].*dt .+ 0.5.*acc_nav.*dt^2
        quat[i,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
    end
    return (pos=pos, vel=vel, quat=quat)
end

function euler_mat(quat)
    n = size(quat,1); E = zeros(n,3)
    for i in 1:n
        e = RotXYZ(QuatRotation(quat[i,1],quat[i,2],quat[i,3],quat[i,4]))
        E[i,:] = [e.theta1, e.theta2, e.theta3]
    end
    return E
end

mkfn(B) = B === nothing ? nothing : (i -> B[i,:])

function test_shape(shape, ml1, sup_v8, rng)
    sd_traj = 3; sd_corr = 103
    scale = shape == :spiral ? 180.0 : (120.0 + 80*rand(rng))
    spec = TrajectorySpec(shape=shape, duration=TEST_DURATION, scale=scale,
                          laps=TEST_LAPS, seed=sd_traj)
    corr = CorruptionSpec(bias_acc=CORR.bias_acc, bias_gyro=CORR.bias_gyro,
                          noise_acc=CORR.noise_acc, noise_gyro=CORR.noise_gyro,
                          rw_acc=CORR.rw_acc, rw_gyro=CORR.rw_gyro, seed=sd_corr)
    r = SyntheticData.generate_dataset([spec], corr; verbose=false)[1]

    p0 = r.pos[1,:]; v0 = r.vel[1,:]; q0 = r.quat[1,:]
    zba = zeros(r.n,3); zbg = zeros(r.n,3)

    # bias-uri ML
    ba1, bg1 = ml1_bias(r, ml1)                                  # ML1
    ba2, bg2 = integrate_bias_v8_calib(r, sup_v8; residual_scale=1.0)  # ML1+ML2

    # configs EKF
    cfg_ref  = EKFClassicConfig()
    cfg_nhc  = EKFNHCConfig()
    cfg_zaru = EKFZARUConfig()

    function ekf_ref(ba,bg)
        p,v,qt,_,_ = run_ekf_classic(r.t,r.acc,r.gyro,p0,v0,q0,r.pos;
                        bias_acc_fn=mkfn(ba), bias_gyro_fn=mkfn(bg), config=cfg_ref)
        return (pos=p, vel=v, quat=qt)
    end
    function ekf_nhc(ba,bg)
        p,v,qt,_,_ = run_ekf_nhc(r.t,r.acc,r.gyro,p0,v0,q0;
                        bias_acc_fn=mkfn(ba), bias_gyro_fn=mkfn(bg), config=cfg_nhc)
        return (pos=p, vel=v, quat=qt)
    end
    function ekf_zaru(ba,bg)
        p,v,qt,_,_ = run_ekf_zaru(r.t,r.acc,r.gyro,p0,v0,q0;
                        bias_acc_fn=mkfn(ba), bias_gyro_fn=mkfn(bg), config=cfg_zaru)
        return (pos=p, vel=v, quat=qt)
    end

    systems = Pair{String,NamedTuple}[]
    push!(systems, "Mecanizare"   => mechanize_state(r, zba, zbg))
    push!(systems, "EKF-ZARU"     => ekf_zaru(nothing,nothing))
    push!(systems, "EKF-NHC"      => ekf_nhc(nothing,nothing))
    push!(systems, "EKF-Ref"      => ekf_ref(nothing,nothing))
    push!(systems, "ML1+Ref"      => ekf_ref(ba1,bg1))
    push!(systems, "ML1+EKF-NHC"  => ekf_nhc(ba1,bg1))
    push!(systems, "ML1+EKF-ZARU" => ekf_zaru(ba1,bg1))
    push!(systems, "ML1+ML2"      => mechanize_state(r, ba2, bg2))
    push!(systems, "ML1+ML2+NHC"  => ekf_nhc(ba2,bg2))
    push!(systems, "ML1+ML2+ZARU" => ekf_zaru(ba2,bg2))

    # referințe pt erori
    Egt = euler_mat(r.quat)
    res = NamedTuple[]
    for (name, st) in systems
        e_pos = traj_rmse(st.pos, r.pos)
        e_vel = traj_rmse(st.vel, r.vel)
        e_ori = traj_rmse(euler_mat(st.quat), Egt)
        push!(res, (name=name, rmse=e_pos, vel=e_vel, ori=e_ori))
    end
    L = traj_length(r.pos)
    return (shape=shape, L=L, res=res)
end

println("="^70)
println("  TEST TOATE SISTEMELE — RMSE + reducere drift vs mecanizare")
println("="^70)

isfile(ML1_FILE) || error("$ML1_FILE nu există.")
isfile(ML2_FILE) || error("$ML2_FILE nu există.")
@load ML1_FILE state;          ml1 = state
@load ML2_FILE supervisor_v8;  sup_v8 = supervisor_v8
println("   ML1 încărcat (window=$(ml1.window))")
println("   ML2 V8 încărcat")

rng = MersenneTwister(42)    

all_results = []
for shape in TEST_SHAPES
    println("\n[*] $shape ...")
    push!(all_results, test_shape(shape, ml1, sup_v8, rng))
end

sysnames = [x.name for x in all_results[1].res]

function print_value_table(title, field, digits)
    println("\n" * "="^100); println("  $title"); println("="^100)
    print("  ", rpad("Sistem", 14))
    for x in all_results; print(rpad(String(x.shape), 11)); end
    println(rpad("MEDIA", 11))
    println("  " * "-"^96)
    for (si, sname) in enumerate(sysnames)
        print("  ", rpad(sname, 14)); vals = Float64[]
        for x in all_results
            v = getfield(x.res[si], field); push!(vals, v)
            print(rpad(string(round(v; digits=digits)), 11))
        end
        println(rpad(string(round(mean(vals); digits=digits)), 11))
    end
    println("="^100)
end

function print_reduction_table(title, field)
    println("\n" * "="^100); println("  $title"); println("="^100)
    print("  ", rpad("Sistem", 14))
    for x in all_results; print(rpad(String(x.shape), 11)); end
    println(rpad("MEDIA", 11))
    println("  " * "-"^96)
    for (si, sname) in enumerate(sysnames)
        print("  ", rpad(sname, 14)); reds = Float64[]
        for x in all_results
            base = getfield(x.res[1], field)         
            red = 100*(base - getfield(x.res[si], field))/max(base,1e-9)
            push!(reds, red); print(rpad(string(round(red; digits=1)), 11))
        end
        println(rpad(string(round(mean(reds); digits=1)), 11))
    end
    println("="^100)
end

print_value_table("RMSE POZIȚIE (m) — per formă", :rmse, 2)
print_reduction_table("REDUCERE DRIFT POZIȚIE vs mecanizare (%)", :rmse)

print_value_table("RMSE VITEZĂ (m/s) — per formă", :vel, 3)
print_reduction_table("REDUCERE EROARE VITEZĂ vs mecanizare (%)", :vel)

print_value_table("RMSE ORIENTARE (rad) — per formă", :ori, 4)
print_reduction_table("REDUCERE EROARE ORIENTARE vs mecanizare (%)", :ori)

println("\n" * "="^100)
println("  SUMAR FINAL — medie pe toate traiectoriile, per metodă")
println("  (RMSE + reducere % față de mecanizare, pentru poziție / viteză / orientare)")
println("="^100)
@printf("  %-14s | %-18s | %-18s | %-18s\n", "Sistem",
        "Poziție (m / %)", "Viteză (m/s / %)", "Orientare (rad / %)")
println("  " * "-"^96)
for (si, sname) in enumerate(sysnames)
    mp = mean(x.res[si].rmse for x in all_results)
    mv = mean(x.res[si].vel  for x in all_results)
    mo = mean(x.res[si].ori  for x in all_results)
    rp = mean(100*(x.res[1].rmse - x.res[si].rmse)/max(x.res[1].rmse,1e-9) for x in all_results)
    rv = mean(100*(x.res[1].vel  - x.res[si].vel )/max(x.res[1].vel, 1e-9) for x in all_results)
    ro = mean(100*(x.res[1].ori  - x.res[si].ori )/max(x.res[1].ori, 1e-9) for x in all_results)
    @printf("  %-14s | %8.2f / %5.1f%% | %8.3f / %5.1f%% | %8.4f / %5.1f%%\n",
            sname, mp, rp, mv, rv, mo, ro)
end
println("="^100)

println("\n✓ Gata. Reducere pozitivă = sistemul bate mecanizarea pură.")
println("  Sumarul final dă driftul mediu redus pe fiecare metodă, pe toate traseele.")