module SyntheticData

using LinearAlgebra
using Random
using Rotations
using Printf

export generate_trajectory, inverse_mechanics, corrupt_imu, generate_dataset,
       TrajectorySpec, CorruptionSpec


const G0 = 9.81 

Base.@kwdef struct TrajectorySpec
    shape::Symbol = :circle      
    duration::Float64 = 60.0     
    fs::Float64 = 100.0         
    scale::Float64 = 5.0         
    laps::Int = 1               
    seed::Int = 0
end


Base.@kwdef struct CorruptionSpec
    bias_acc::Vector{Float64}  = [0.3, -0.2, 0.15]     # m/s²
    bias_gyro::Vector{Float64} = [0.02, -0.015, 0.01]  # rad/s
    
    # Zgomot alb
    noise_acc::Float64  = 0.05    # m/s² std
    noise_gyro::Float64 = 0.002   # rad/s std
    
    # Random walk
    rw_acc::Float64  = 0.01       # m/s² per √s
    rw_gyro::Float64 = 0.0005     # rad/s per √s
    seed::Int = 0
end

function rpy_to_R(roll, pitch, yaw)
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    Rz = [cy -sy 0; sy cy 0; 0 0 1]
    Ry = [cp 0 sp; 0 1 0; -sp 0 cp]
    Rx = [1 0 0; 0 cr -sr; 0 sr cr]
    return Rz * Ry * Rx
end

function rpy_to_quat(roll, pitch, yaw)
    cr,sr = cos(roll/2), sin(roll/2)
    cp,sp = cos(pitch/2), sin(pitch/2)
    cy,sy = cos(yaw/2), sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return [w,x,y,z]
end


function generate_trajectory(spec::TrajectorySpec)
    rng = MersenneTwister(spec.seed)
    n_per_lap = round(Int, spec.duration * spec.fs)
    n = n_per_lap * spec.laps
    dt = 1.0 / spec.fs
    t = collect(0:n-1) .* dt

    pos = zeros(n, 3)
    s = spec.scale
    Dlap = spec.duration 

    if spec.shape == :circle
        ω = 2π / Dlap * 2   
        for i in 1:n
            tl = t[i]   
            pos[i,1] = s*cos(ω*tl)
            pos[i,2] = s*sin(ω*tl)
            pos[i,3] = 0.5*sin(0.5*ω*tl)
        end
    elseif spec.shape == :figure8
        ω = 2π / Dlap * 2
        for i in 1:n
            tl = t[i]
            pos[i,1] = s*sin(ω*tl)
            pos[i,2] = s*sin(ω*tl)*cos(ω*tl)
            pos[i,3] = 0.3*sin(ω*tl)
        end
    elseif spec.shape == :spiral
        ω = 2π / Dlap * 3
        for i in 1:n
            tl = mod(t[i], Dlap)
            r = s * tl/Dlap
            pos[i,1] = r*cos(ω*tl)
            pos[i,2] = r*sin(ω*tl)
            pos[i,3] = 0.5*tl/Dlap
        end
    elseif spec.shape == :rectangle
        per = n_per_lap ÷ 4
        for i in 1:n
            il = ((i-1) % n_per_lap)
            ph = il ÷ per
            fr = (il % per) / per
            if ph == 0
                pos[i,1]=s*(2fr-1); pos[i,2]=-s
            elseif ph == 1
                pos[i,1]=s; pos[i,2]=s*(2fr-1)
            elseif ph == 2
                pos[i,1]=s*(1-2fr); pos[i,2]=s
            else
                pos[i,1]=-s; pos[i,2]=s*(1-2fr)
            end
            pos[i,3]=0.0
        end
    elseif spec.shape == :random
        K = 5  
        Ttot = Dlap * spec.laps
        freqs = 0.5 .+ 2.5 .* rand(rng, K, 2)        
        amps  = (s/K) .* (0.5 .+ 0.5 .* rand(rng, K, 2))
        phases = 2π .* rand(rng, K, 2)
        for i in 1:n
            tl = t[i]
            x = 0.0; y = 0.0
            for k in 1:K
                x += amps[k,1] * sin(2π * freqs[k,1] * tl/Ttot + phases[k,1])
                y += amps[k,2] * sin(2π * freqs[k,2] * tl/Ttot + phases[k,2])
            end
            pos[i,1] = x; pos[i,2] = y
            pos[i,3] = 0.3 * sin(2π * tl/Ttot)
        end
    else
        step = s/spec.fs
        v = zeros(3)
        for i in 2:n
            v .+= step .* (randn(rng,3) .- 0.0)
            v .*= 0.98  
            pos[i,:] = pos[i-1,:] .+ v .* dt .* spec.fs
        end
    end

    vel = zeros(n,3); acc_world = zeros(n,3)
    for i in 2:n-1
        vel[i,:] = (pos[i+1,:] .- pos[i-1,:]) ./ (2dt)
    end
    vel[1,:]=vel[2,:]; vel[n,:]=vel[n-1,:]
    for i in 2:n-1
        acc_world[i,:] = (vel[i+1,:] .- vel[i-1,:]) ./ (2dt)
    end
    acc_world[1,:]=acc_world[2,:]; acc_world[n,:]=acc_world[n-1,:]

    rpy = zeros(n,3)
    quat = zeros(n,4)   # [w,x,y,z]
    for i in 1:n
        yaw = atan(vel[i,2], vel[i,1])
        roll  = 0.1*sin(0.5*t[i])
        pitch = 0.1*cos(0.3*t[i])
        rpy[i,:] = [roll, pitch, yaw]
        R = RotXYZ(roll, pitch, yaw)    
        q = QuatRotation(R)
        quat[i,:] = [q.q.s, q.q.v1, q.q.v2, q.q.v3] 
    end

    return t, pos, vel, acc_world, rpy, quat
end


function inverse_mechanics(t, acc_world, quat)
    n = size(acc_world,1)
    gyro_ideal = zeros(n,3)
    acc_ideal  = zeros(n,3)
    g_world = [0.0, 0.0, -G0]

    for i in 1:n
        q = QuatRotation(quat[i,1], quat[i,2], quat[i,3], quat[i,4])  
        acc_ideal[i,:] = inv(q) * (acc_world[i,:] .- g_world)
    end

    for i in 2:n
        dt = t[i]-t[i-1]; dt = dt<=0 ? 0.01 : dt
        qprev = QuatRotation(quat[i-1,1], quat[i-1,2], quat[i-1,3], quat[i-1,4])
        qcur  = QuatRotation(quat[i,1],   quat[i,2],   quat[i,3],   quat[i,4])
        dq = inv(qprev) * qcur
        rxyz = RotXYZ(dq)                  
        gyro_ideal[i,:] = [rxyz.theta1/dt, rxyz.theta2/dt, rxyz.theta3/dt]
    end
    gyro_ideal[1,:] = gyro_ideal[2,:]

    return gyro_ideal, acc_ideal
end


function corrupt_imu(gyro_ideal, acc_ideal, fs, corr::CorruptionSpec)
    rng = MersenneTwister(corr.seed)
    n = size(acc_ideal,1)
    dt = 1.0/fs

    gyro_c = similar(gyro_ideal)
    acc_c  = similar(acc_ideal)
    bias_acc_true  = zeros(n,3)   
    bias_gyro_true = zeros(n,3)  
    bias_acc_det   = zeros(n,3)   
    bias_gyro_det  = zeros(n,3)

    rw_a = zeros(3); rw_g = zeros(3)
    for i in 1:n
        rw_a .+= corr.rw_acc  * sqrt(dt) .* randn(rng,3)
        rw_g .+= corr.rw_gyro * sqrt(dt) .* randn(rng,3)
        ba_det = corr.bias_acc  .+ rw_a
        bg_det = corr.bias_gyro .+ rw_g
        na = corr.noise_acc  .* randn(rng,3)
        ng = corr.noise_gyro .* randn(rng,3)
        acc_c[i,:]  = acc_ideal[i,:]  .+ ba_det .+ na
        gyro_c[i,:] = gyro_ideal[i,:] .+ bg_det .+ ng
        bias_acc_true[i,:]  = ba_det .+ na
        bias_gyro_true[i,:] = bg_det .+ ng
        bias_acc_det[i,:]   = ba_det
        bias_gyro_det[i,:]  = bg_det
    end

    return gyro_c, acc_c, bias_acc_true, bias_gyro_true, bias_acc_det, bias_gyro_det
end


function generate_dataset(traj_specs::Vector{TrajectorySpec}, corr::CorruptionSpec; verbose=true)
    runs = []
    for (k, spec) in enumerate(traj_specs)
        t, pos, vel, acc_w, rpy, quat = generate_trajectory(spec)
        gyro_id, acc_id = inverse_mechanics(t, acc_w, quat)
        corr_k = CorruptionSpec(
            bias_acc=corr.bias_acc, bias_gyro=corr.bias_gyro,
            noise_acc=corr.noise_acc, noise_gyro=corr.noise_gyro,
            rw_acc=corr.rw_acc, rw_gyro=corr.rw_gyro, seed=corr.seed+k)
        gyro_c, acc_c, ba_true, bg_true, ba_det, bg_det = corrupt_imu(gyro_id, acc_id, spec.fs, corr_k)

        push!(runs, (
            t=t, pos=pos, vel=vel, quat=quat, rpy=rpy,
            gyro_ideal=gyro_id, acc_ideal=acc_id,
            gyro=gyro_c, acc=acc_c,
            bias_acc_true=ba_true, bias_gyro_true=bg_true,      
            bias_acc_det=ba_det, bias_gyro_det=bg_det,         
            shape=spec.shape, n=length(t), fs=spec.fs,
        ))
        verbose && println("   ✓ run $k ($(spec.shape) ×$(spec.laps) laps, $(length(t)) eșantioane)")
    end
    return runs
end


function save_run_csv(run, dir::String, name::String)
    isdir(dir) || mkpath(dir)
    open(joinpath(dir, "imu_$(name).csv"), "w") do io
        println(io, "time,gyro_x,gyro_y,gyro_z,acc_x,acc_y,acc_z,bias_acc_x,bias_acc_y,bias_acc_z,bias_gyro_x,bias_gyro_y,bias_gyro_z")
        for i in 1:run.n
            @printf(io, "%.6f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                run.t[i], run.gyro[i,1],run.gyro[i,2],run.gyro[i,3],
                run.acc[i,1],run.acc[i,2],run.acc[i,3],
                run.bias_acc_true[i,1],run.bias_acc_true[i,2],run.bias_acc_true[i,3],
                run.bias_gyro_true[i,1],run.bias_gyro_true[i,2],run.bias_gyro_true[i,3])
        end
    end
    open(joinpath(dir, "gt_$(name).csv"), "w") do io
        println(io, "time,x,y,z,qw,qx,qy,qz")
        for i in 1:run.n
            @printf(io, "%.6f,%.6f,%.6f,%.6f,%.8f,%.8f,%.8f,%.8f\n",
                run.t[i], run.pos[i,1],run.pos[i,2],run.pos[i,3],
                run.quat[i,1],run.quat[i,2],run.quat[i,3],run.quat[i,4])
        end
    end
end


function save_dataset_csv(runs, dir::String; prefix::String="run")
    isdir(dir) || mkpath(dir)
    for (k, r) in enumerate(runs)
        save_run_csv(r, dir, "$(prefix)_$(k)_$(r.shape)")
    end
    println("   ✓ $(length(runs)) runs salvate în $dir/")
end

export save_run_csv, save_dataset_csv

end 