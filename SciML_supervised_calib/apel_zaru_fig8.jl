using Pkg
Pkg.activate(".")


include("SyntheticData.jl")
include("EKF_classic.jl")
using .SyntheticData
using .EKFClassic
using LinearAlgebra
using Rotations
using Statistics
using Printf
using Plots

traj_rmse(p,gt) = (n=min(size(p,1),size(gt,1)); sqrt(mean(sum((p[1:n,:].-gt[1:n,:]).^2,dims=2))))

function euler_matrix(quat)
    n=size(quat,1); E=zeros(n,3)
    for i in 1:n
        e=RotXYZ(QuatRotation(quat[i,1],quat[i,2],quat[i,3],quat[i,4]))
        E[i,1]=e.theta1; E[i,2]=e.theta2; E[i,3]=e.theta3
    end
    return E
end

function mechanize(t, acc, gyro, p0, v0, q0)
    n = length(t); g_world = [0.0,0.0,-9.81]
    pos = zeros(n,3); vel = zeros(n,3); quat = zeros(n,4)
    q = QuatRotation(q0[1],q0[2],q0[3],q0[4])
    pos[1,:]=p0; vel[1,:]=v0; quat[1,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
    for i in 2:n
        dt=t[i]-t[i-1]; dt=dt<=0 ? 0.01 : dt
        dθ = gyro[i,:].*dt
        q = q*QuatRotation(RotXYZ(dθ[1],dθ[2],dθ[3]))
        acc_nav = q*acc[i,:].+g_world
        vel[i,:]=vel[i-1,:].+acc_nav.*dt
        pos[i,:]=pos[i-1,:].+vel[i-1,:].*dt.+0.5.*acc_nav.*dt^2
        quat[i,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
    end
    return pos, vel, quat
end

function save_axis_figures(outdir, t, series)
    an = ["X","Y","Z"]; rpy_names = ["Roll","Pitch","Yaw"]

    po = plot(layout=(3,1), size=(850,900), legend=:topright,
              plot_title="Orientare — figure8 (ZARU vs mecanizare pură)")
    for ax in 1:3, s in series
        E = euler_matrix(s.quat)
        plot!(po[ax], t, E[:,ax], label=(ax==1 ? s.label : ""),
              lw=s.lw, color=s.color, ls=s.ls, alpha=s.alpha,
              ylabel="$(rpy_names[ax]) (rad)", xlabel=(ax==3 ? "Timp (s)" : ""))
    end
    savefig(po, joinpath(outdir, "orientare_rpy.png"))

    pp = plot(layout=(3,1), size=(850,900), legend=:topright,
              plot_title="Poziție — figure8 (ZARU vs mecanizare pură)")
    for ax in 1:3, s in series
        plot!(pp[ax], t, s.pos[:,ax], label=(ax==1 ? s.label : ""),
              lw=s.lw, color=s.color, ls=s.ls, alpha=s.alpha,
              ylabel="poziție $(an[ax]) (m)", xlabel=(ax==3 ? "Timp (s)" : ""))
    end
    savefig(pp, joinpath(outdir, "pozitie_xyz.png"))

    pv = plot(layout=(3,1), size=(850,900), legend=:topright,
              plot_title="Viteză — figure8 (ZARU vs mecanizare pură)")
    for ax in 1:3, s in series
        plot!(pv[ax], t, s.vel[:,ax], label=(ax==1 ? s.label : ""),
              lw=s.lw, color=s.color, ls=s.ls, alpha=s.alpha,
              ylabel="viteză $(an[ax]) (m/s)", xlabel=(ax==3 ? "Timp (s)" : ""))
    end
    savefig(pv, joinpath(outdir, "viteza_xyz.png"))
end

spec = TrajectorySpec(shape=:figure8, duration=25, scale=150, laps=4, seed=3)
corr = CorruptionSpec(
    bias_acc=[0.1,-0.08,0.06], bias_gyro=[0.01,-0.008,0.006],
    noise_acc=0.03, noise_gyro=0.0015, rw_acc=0.005, rw_gyro=0.0003, seed=103)

t, pos, vel, acc_w, rpy, quat = generate_trajectory(spec)
gyro_id, acc_id = inverse_mechanics(t, acc_w, quat)
gyro_c, acc_c, _, _, _, _ = corrupt_imu(gyro_id, acc_id, spec.fs, corr)
p0 = pos[1,:]; v0 = vel[1,:]; q0 = quat[1,:]

OUT = "results_zaru_figure8"; isdir(OUT) || mkpath(OUT)

println("="^64)
println("  EKF-ZARU vs MECANIZARE PURĂ — figure8 ($(spec.duration*spec.laps)s)")
println("="^64)

absz = abs.(gyro_id[:,3])
@printf("\n  Yaw-rate |gyro_z| (ideal): min=%.4f  median=%.4f  max=%.4f rad/s\n",
        minimum(absz), median(absz), maximum(absz))
for thr in (0.02, 0.05, 0.1, 0.2)
    frac = 100*count(<(thr), absz)/length(absz)
    @printf("    prag %.2f → %.1f%% din traseu are yaw-rate sub prag\n", thr, frac)
end
println("  (alege gyro_still_thresh între min și median pt a prinde minimele fără vârfuri)")

cfg_zaru = EKFZARUConfig(gyro_still_thresh=0.05, bias_lr=0.02, sigma_acc_proc=0.3)
posZ, velZ, quatZ, _, obsZ = run_ekf_zaru(t, acc_c, gyro_c, p0, v0, q0; config=cfg_zaru)
rmseZ = traj_rmse(posZ, pos)

posM, velM, quatM = mechanize(t, acc_c, gyro_c, p0, v0, q0)
rmseM = traj_rmse(posM, pos)

@printf("\n  Mecanizare              : RMSE = %.2f m\n", rmseM)
@printf("  EKF-ZARU autonom             : RMSE = %.2f m  (%d faze rotație-mică)\n", rmseZ, length(obsZ))
@printf("  Fracție traseu în faze still : %.1f%%\n", 100*length(obsZ)/length(t))
@printf("  Reducere derivă ZARU vs pură : %.1f%%\n", 100*(rmseM-rmseZ)/max(rmseM,1e-9))

p2d = plot(pos[:,1], pos[:,2], label="referință (GT)", lw=3, color=:black,
           xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal, legend=:best,
           title="Fundă — DR=$(round(rmseM,digits=1)) m | ZARU=$(round(rmseZ,digits=1)) m")
plot!(p2d, posM[:,1], posM[:,2], label="mecanizare", lw=2, color=:red, ls=:dot)
plot!(p2d, posZ[:,1], posZ[:,2], label="EKF-ZARU (autonom)", lw=2, color=:mediumpurple)
savefig(p2d, joinpath(OUT, "traj_figure8.png"))

series = [
    (label="referință", color=:black, ls=:solid, lw=2.5, alpha=1.0, pos=pos, vel=vel, quat=quat),
    (label="mecanizare", color=:red, ls=:dot, lw=1.3, alpha=0.8, pos=posM, vel=velM, quat=quatM),
    (label="EKF-ZARU", color=:mediumpurple, ls=:solid, lw=1.5, alpha=0.85, pos=posZ, vel=velZ, quat=quatZ),
]
save_axis_figures(OUT, t, series)

println("\n✓ Grafice salvate în $OUT/")
println("   traj_figure8.png   — traiectorie 2D (XY)")
println("   orientare_rpy.png  — roll / pitch / yaw")
println("   pozitie_xyz.png    — poziție X / Y / Z")
println("   viteza_xyz.png     — viteză X / Y / Z")
