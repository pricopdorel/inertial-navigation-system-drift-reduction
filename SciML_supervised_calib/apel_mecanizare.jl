using Pkg
Pkg.activate(".")


include("SyntheticData.jl")
using .SyntheticData
using LinearAlgebra
using Rotations
using Statistics
using Printf
using Plots

function mechanize(t, acc, gyro, p0, v0, q0; ba=nothing, bg=nothing)
    n = length(t)
    g_world = [0.0, 0.0, -9.81]
    pos = zeros(n,3); vel = zeros(n,3); quat = zeros(n,4)
    q = QuatRotation(q0[1], q0[2], q0[3], q0[4])
    pos[1,:] = p0; vel[1,:] = v0; quat[1,:] = [q.q.s, q.q.v1, q.q.v2, q.q.v3]
    for i in 2:n
        dt = t[i]-t[i-1]; dt = dt<=0 ? 0.01 : dt
        gyro_i = gyro[i,:]; acc_i = acc[i,:]
        bg !== nothing && (gyro_i = gyro_i .- bg[i,:])
        ba !== nothing && (acc_i  = acc_i  .- ba[i,:])
        dθ = gyro_i .* dt
        q = q * QuatRotation(RotXYZ(dθ[1], dθ[2], dθ[3]))
        acc_nav = q * acc_i .+ g_world
        vel[i,:] = vel[i-1,:] .+ acc_nav .* dt
        pos[i,:] = pos[i-1,:] .+ vel[i-1,:] .* dt .+ 0.5 .* acc_nav .* dt^2
        quat[i,:] = [q.q.s, q.q.v1, q.q.v2, q.q.v3]
    end
    return pos, vel, quat
end

traj_rmse(p, gt) = (n=min(size(p,1),size(gt,1)); sqrt(mean(sum((p[1:n,:].-gt[1:n,:]).^2, dims=2))))

function euler_matrix(quat)
    n = size(quat,1); E = zeros(n,3)
    for i in 1:n
        e = RotXYZ(QuatRotation(quat[i,1], quat[i,2], quat[i,3], quat[i,4]))
        E[i,1]=e.theta1; E[i,2]=e.theta2; E[i,3]=e.theta3
    end
    return E
end

spec = TrajectorySpec(shape=:circle, duration=25, scale=150, laps=4, seed=1)
corr = CorruptionSpec(
    bias_acc=[0.1,-0.08,0.06], bias_gyro=[0.01,-0.008,0.006],
    noise_acc=0.03, noise_gyro=0.0015, rw_acc=0.005, rw_gyro=0.0003, seed=101)

t, pos, vel, acc_w, rpy, quat = generate_trajectory(spec)
gyro_id, acc_id = inverse_mechanics(t, acc_w, quat)
gyro_c, acc_c, _, _, _, _ = corrupt_imu(gyro_id, acc_id, spec.fs, corr)

p0 = pos[1,:]; v0 = vel[1,:]; q0 = quat[1,:]

println("="^60)
println("  MECANIZARE pe cerc sintetic (scale=$(spec.scale), $(spec.duration)s × $(spec.laps) laps = $(spec.duration*spec.laps)s)")
println("="^60)

pos_id, _, _ = mechanize(t, acc_id, gyro_id, p0, v0, q0)
rmse_id = traj_rmse(pos_id, pos)
@printf("  [1] Mecanizare pe date IDEALE   → RMSE vs referință = %.4e m\n", rmse_id)
println("      (trebuie aproape zero — confirmă consistența generator↔mecanizare)")

pos_pur, vel_pur, quat_pur = mechanize(t, acc_c, gyro_c, p0, v0, q0)
rmse_pur = traj_rmse(pos_pur, pos)
@printf("  [2] Mecanizare pe date PERTURBATE (pură) → RMSE vs referință = %.2f m\n", rmse_pur)
println("      (derivă mare — justifică nevoia corecției de bias)")

isdir("results_mechanization") || mkpath("results_mechanization")
p = plot(pos[:,1], pos[:,2], label="referință (GT)", lw=3, color=:black,
         xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal, legend=:best,
         title="Mecanizare cerc — ideal vs pură")
plot!(p, pos_id[:,1], pos_id[:,2], label="mecanizare ideală (RMSE=$(round(rmse_id,sigdigits=2)))",
      lw=2, color=:green, ls=:dash)
plot!(p, pos_pur[:,1], pos_pur[:,2], label="mecanizare pură (RMSE=$(round(rmse_pur,digits=1)))",
      lw=2, color=:red)
savefig(p, "results_mechanization/mecanizare_cerc.png")

an = ["X","Y","Z"]
rpy_names = ["Roll","Pitch","Yaw"]
E_gt  = euler_matrix(quat)       # orientarea de referință
E_pur = euler_matrix(quat_pur)   # orientarea mecanizată

po = plot(layout=(3,1), size=(850,900), legend=:topright,
          plot_title="Orientare — referință vs mecanizare pură (cerc)")
for ax in 1:3
    plot!(po[ax], t, E_gt[:,ax], label=(ax==1 ? "referință" : ""),
          lw=2.5, color=:black, ylabel="$(rpy_names[ax]) (rad)",
          xlabel=(ax==3 ? "Timp (s)" : ""))
    plot!(po[ax], t, E_pur[:,ax], label=(ax==1 ? "mecanizare pură" : ""),
          lw=1.5, color=:red, alpha=0.85)
end
savefig(po, "results_mechanization/orientare_rpy.png")

pp = plot(layout=(3,1), size=(850,900), legend=:topright,
          plot_title="Poziție — referință vs mecanizare pură (cerc)")
for ax in 1:3
    plot!(pp[ax], t, pos[:,ax], label=(ax==1 ? "referință" : ""),
          lw=2.5, color=:black, ylabel="poziție $(an[ax]) (m)",
          xlabel=(ax==3 ? "Timp (s)" : ""))
    plot!(pp[ax], t, pos_pur[:,ax], label=(ax==1 ? "mecanizare pură" : ""),
          lw=1.5, color=:red, alpha=0.85)
end
savefig(pp, "results_mechanization/pozitie_xyz.png")

pv = plot(layout=(3,1), size=(850,900), legend=:topright,
          plot_title="Viteză — referință vs mecanizare pură (cerc)")
for ax in 1:3
    plot!(pv[ax], t, vel[:,ax], label=(ax==1 ? "referință" : ""),
          lw=2.5, color=:black, ylabel="viteză $(an[ax]) (m/s)",
          xlabel=(ax==3 ? "Timp (s)" : ""))
    plot!(pv[ax], t, vel_pur[:,ax], label=(ax==1 ? "mecanizare pură" : ""),
          lw=1.5, color=:red, alpha=0.85)
end
savefig(pv, "results_mechanization/viteza_xyz.png")

println("\n✓ Grafice salvate în results_mechanization/")
println("   mecanizare_cerc.png  — traiectorie 2D (XY)")
println("   orientare_rpy.png    — roll / pitch / yaw")
println("   pozitie_xyz.png      — poziție X / Y / Z")
println("   viteza_xyz.png       — viteză X / Y / Z")