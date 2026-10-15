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

function save_axis_figures(outdir, t, series)
    an = ["X","Y","Z"]; rpy_names = ["Roll","Pitch","Yaw"]
    po = plot(layout=(3,1), size=(850,900), legend=:topright,
              plot_title="Orientare — rectangle (NHC vs EKF-GPS)")
    for ax in 1:3, s in series
        E = euler_matrix(s.quat)
        plot!(po[ax], t, E[:,ax], label=(ax==1 ? s.label : ""),
              lw=s.lw, color=s.color, ls=s.ls, alpha=s.alpha,
              ylabel="$(rpy_names[ax]) (rad)", xlabel=(ax==3 ? "Timp (s)" : ""))
    end
    savefig(po, joinpath(outdir, "orientare_rpy.png"))

    pp = plot(layout=(3,1), size=(850,900), legend=:topright,
              plot_title="Poziție — rectangle (NHC vs EKF-GPS)")
    for ax in 1:3, s in series
        plot!(pp[ax], t, s.pos[:,ax], label=(ax==1 ? s.label : ""),
              lw=s.lw, color=s.color, ls=s.ls, alpha=s.alpha,
              ylabel="poziție $(an[ax]) (m)", xlabel=(ax==3 ? "Timp (s)" : ""))
    end
    savefig(pp, joinpath(outdir, "pozitie_xyz.png"))

    pv = plot(layout=(3,1), size=(850,900), legend=:topright,
              plot_title="Viteză — rectangle (NHC vs EKF-GPS)")
    for ax in 1:3, s in series
        plot!(pv[ax], t, s.vel[:,ax], label=(ax==1 ? s.label : ""),
              lw=s.lw, color=s.color, ls=s.ls, alpha=s.alpha,
              ylabel="viteză $(an[ax]) (m/s)", xlabel=(ax==3 ? "Timp (s)" : ""))
    end
    savefig(pv, joinpath(outdir, "viteza_xyz.png"))
end

const DLAP = 25.0
spec = TrajectorySpec(shape=:rectangle, duration=DLAP, scale=150, laps=4, seed=5)
corr = CorruptionSpec(
    bias_acc=[0.1,-0.08,0.06], bias_gyro=[0.01,-0.008,0.006],
    noise_acc=0.03, noise_gyro=0.0015, rw_acc=0.005, rw_gyro=0.0003, seed=105)

t, pos, vel, acc_w, rpy, quat = generate_trajectory(spec)
gyro_id, acc_id = inverse_mechanics(t, acc_w, quat)
gyro_c, acc_c, _, _, _, _ = corrupt_imu(gyro_id, acc_id, spec.fs, corr)
p0 = pos[1,:]; v0 = vel[1,:]; q0 = quat[1,:]

OUT = "results_nhc_vs_gps_rectangle"; isdir(OUT) || mkpath(OUT)

println("="^66)
println("  RECTANGLE ($(spec.duration*spec.laps)s) — NHC autonom vs EKF-GPS")
println("="^66)

cfg_nhc = EKFNHCConfig(sigma_nhc=0.05, nhc_period_s=0.1, sigma_acc_proc=0.3)
posN, velN, quatN, _, obsN = run_ekf_nhc(t, acc_c, gyro_c, p0, v0, q0; config=cfg_nhc)
rmseN = traj_rmse(posN, pos)

cfg_gps = EKFClassicConfig(obs_period_s=1.0, obs_velocity=true,
                           sigma_pos_meas=0.1, sigma_vel_meas=0.1, sigma_acc_proc=0.3)
posG, velG, quatG, _, obsG = run_ekf_classic(
    t, acc_c, gyro_c, p0, v0, q0, pos; ref_vel=vel, config=cfg_gps)
rmseG = traj_rmse(posG, pos)

@printf("\n  EKF-NHC autonom        : RMSE = %.2f m  (fără referință, %d pseudo-obs)\n", rmseN, length(obsN))
@printf("  EKF cu GPS (referință) : RMSE = %.2f m  (%d achiziții la 1 Hz)\n", rmseG, length(obsG))
@printf("\n  GPS vs NHC : %+.1f%% (cât câștigă referința față de autonom)\n",
        100*(rmseN-rmseG)/max(rmseN,1e-9))

p2d = plot(pos[:,1], pos[:,2], label="referință (GT)", lw=3, color=:black,
           xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal, legend=:best,
           title="Dreptunghi — NHC=$(round(rmseN,digits=0)) m | GPS=$(round(rmseG,digits=1)) m")
plot!(p2d, posN[:,1], posN[:,2], label="EKF-NHC (autonom)", lw=2, color=:mediumpurple)
plot!(p2d, posG[:,1], posG[:,2], label="EKF-GPS (referință)", lw=2, color=:dodgerblue)
savefig(p2d, joinpath(OUT, "traj_rectangle.png"))

series = [
    (label="referință", color=:black, ls=:solid, lw=2.5, alpha=1.0, pos=pos, vel=vel, quat=quat),
    (label="EKF-NHC autonom", color=:mediumpurple, ls=:solid, lw=1.4, alpha=0.85, pos=posN, vel=velN, quat=quatN),
    (label="EKF-GPS", color=:dodgerblue, ls=:solid, lw=1.5, alpha=0.85, pos=posG, vel=velG, quat=quatG),
]
save_axis_figures(OUT, t, series)

println("\n✓ Grafice salvate în $OUT/")
println("   traj_rectangle.png, orientare_rpy.png, pozitie_xyz.png, viteza_xyz.png")
