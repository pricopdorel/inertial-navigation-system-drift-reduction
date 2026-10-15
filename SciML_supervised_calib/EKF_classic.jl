module EKFClassic

using LinearAlgebra
using Rotations

export run_ekf_classic, EKFClassicConfig, run_ekf_nhc, EKFNHCConfig, run_ekf_zaru, EKFZARUConfig

Base.@kwdef struct EKFClassicConfig
    sigma_acc_proc::Float64 = 0.3     
    sigma_pos_meas::Float64 = 0.1      
    sigma_vel_meas::Float64 = 0.1      
    sigma_p0::Float64 = 1.0           
    sigma_v0::Float64 = 0.5            
    obs_period_s::Float64 = 1.0        
    obs_velocity::Bool = true         
    gravity::Vector{Float64} = [0.0,0.0,-9.81]
end

function run_ekf_classic(t, acc, gyro, init_p, init_v, init_q, ref_pos;
                         ref_vel=nothing,
                         bias_acc_fn=nothing, bias_gyro_fn=nothing,
                         config::EKFClassicConfig=EKFClassicConfig())
    n = length(t)
    x = vcat(Vector{Float64}(init_p), Vector{Float64}(init_v))   
    q = QuatRotation(init_q[1], init_q[2], init_q[3], init_q[4])

    P = Matrix(Diagonal(vcat(fill(config.sigma_p0^2,3), fill(config.sigma_v0^2,3))))

    use_vel = config.obs_velocity && (ref_vel !== nothing)
    if use_vel
        H = Matrix{Float64}(I, 6, 6)
        R = Matrix(Diagonal(vcat(fill(config.sigma_pos_meas^2,3),
                                 fill(config.sigma_vel_meas^2,3))))
    else
        H = zeros(3,6); H[1:3,1:3] = I(3)
        R = Matrix(Diagonal(fill(config.sigma_pos_meas^2, 3)))
    end

    est_pos = zeros(n,3); est_vel = zeros(n,3); est_quat = zeros(n,4)
    P_trace = zeros(n); obs_times = Int[]
    est_pos[1,:]=x[1:3]; est_vel[1,:]=x[4:6]
    est_quat[1,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
    P_trace[1] = tr(P[1:3,1:3])

    dt_mean = n > 1 ? (t[end]-t[1])/(n-1) : 0.01
    obs_step = max(1, round(Int, config.obs_period_s / dt_mean))
    last_obs = 1

    for i in 2:n
        dt = t[i]-t[i-1]; dt = dt<=0 ? 0.01 : dt

        gyro_i = Vector{Float64}(gyro[i,:])
        acc_i  = Vector{Float64}(acc[i,:])
        if bias_gyro_fn !== nothing; gyro_i = gyro_i .- bias_gyro_fn(i); end
        if bias_acc_fn  !== nothing; acc_i  = acc_i  .- bias_acc_fn(i);  end

        dθ = gyro_i .* dt
        q = q * QuatRotation(RotXYZ(dθ[1],dθ[2],dθ[3]))

        acc_nav = q * acc_i .+ config.gravity
        x[1:3] = x[1:3] .+ x[4:6].*dt .+ 0.5.*acc_nav.*dt^2
        x[4:6] = x[4:6] .+ acc_nav.*dt

        F = Matrix{Float64}(I,6,6); F[1:3,4:6] = I(3).*dt
        Q = zeros(6,6)
        Q[4:6,4:6] = I(3) .* (config.sigma_acc_proc^2 * dt)
        Q[1:3,1:3] = I(3) .* (0.25*config.sigma_acc_proc^2 * dt^3)
        P = F*P*F' + Q
        P = 0.5*(P+P')

        if (i - last_obs) >= obs_step
            if use_vel
                z = vcat(Vector{Float64}(ref_pos[i,:]), Vector{Float64}(ref_vel[i,:]))
                y = z .- x                      
            else
                z = Vector{Float64}(ref_pos[i,:])
                y = z .- x[1:3]                
            end
            S = H*P*H' + R                      
            K = (P*H') / S                       
            x = x .+ K*y                         
            P = (I(6) - K*H)*P                   
            P = 0.5*(P+P')
            last_obs = i
            push!(obs_times, i)
        end

        est_pos[i,:]=x[1:3]; est_vel[i,:]=x[4:6]
        est_quat[i,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
        P_trace[i] = tr(P[1:3,1:3])
    end

    return est_pos, est_vel, est_quat, P_trace, obs_times
end


Base.@kwdef struct EKFNHCConfig
    sigma_acc_proc::Float64 = 0.3      
    sigma_nhc::Float64 = 0.05          
    sigma_p0::Float64 = 1.0
    sigma_v0::Float64 = 0.5
    nhc_period_s::Float64 = 0.1         
    gravity::Vector{Float64} = [0.0,0.0,-9.81]
end


function run_ekf_nhc(t, acc, gyro, init_p, init_v, init_q;
                     bias_acc_fn=nothing, bias_gyro_fn=nothing,
                     config::EKFNHCConfig=EKFNHCConfig())
    n = length(t)
    x = vcat(Vector{Float64}(init_p), Vector{Float64}(init_v))
    q = QuatRotation(init_q[1], init_q[2], init_q[3], init_q[4])
    P = Matrix(Diagonal(vcat(fill(config.sigma_p0^2,3), fill(config.sigma_v0^2,3))))

    R = Matrix(Diagonal(fill(config.sigma_nhc^2, 2)))

    est_pos=zeros(n,3); est_vel=zeros(n,3); est_quat=zeros(n,4)
    P_trace=zeros(n); obs_times=Int[]
    est_pos[1,:]=x[1:3]; est_vel[1,:]=x[4:6]
    est_quat[1,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]; P_trace[1]=tr(P[1:3,1:3])

    dt_mean = n>1 ? (t[end]-t[1])/(n-1) : 0.01
    nhc_step = max(1, round(Int, config.nhc_period_s/dt_mean))
    last_obs = 1

    for i in 2:n
        dt = t[i]-t[i-1]; dt = dt<=0 ? 0.01 : dt

        gyro_i = Vector{Float64}(gyro[i,:]); acc_i = Vector{Float64}(acc[i,:])
        bias_gyro_fn !== nothing && (gyro_i = gyro_i .- bias_gyro_fn(i))
        bias_acc_fn  !== nothing && (acc_i  = acc_i  .- bias_acc_fn(i))

        dθ = gyro_i .* dt
        q = q * QuatRotation(RotXYZ(dθ[1],dθ[2],dθ[3]))

        acc_nav = q * acc_i .+ config.gravity
        x[1:3] = x[1:3] .+ x[4:6].*dt .+ 0.5.*acc_nav.*dt^2
        x[4:6] = x[4:6] .+ acc_nav.*dt

        F = Matrix{Float64}(I,6,6); F[1:3,4:6] = I(3).*dt
        Q = zeros(6,6)
        Q[4:6,4:6] = I(3) .* (config.sigma_acc_proc^2 * dt)
        Q[1:3,1:3] = I(3) .* (0.25*config.sigma_acc_proc^2 * dt^3)
        P = F*P*F' + Q; P = 0.5*(P+P')

        if (i - last_obs) >= nhc_step
            Rmat = Matrix(q)                
            RT = Rmat'                       
            v_body = RT * x[4:6]              
            H = zeros(2,6)
            H[1,4:6] = RT[2,:]               
            H[2,4:6] = RT[3,:]               
            z = [0.0, 0.0]                     
            y = z .- [v_body[2], v_body[3]]  
            S = H*P*H' + R
            K = (P*H') / S
            x = x .+ K*y                      
            P = (I(6) - K*H)*P; P = 0.5*(P+P')
            last_obs = i
            push!(obs_times, i)
        end

        est_pos[i,:]=x[1:3]; est_vel[i,:]=x[4:6]
        est_quat[i,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
        P_trace[i]=tr(P[1:3,1:3])
    end

    return est_pos, est_vel, est_quat, P_trace, obs_times
end

Base.@kwdef struct EKFZARUConfig
    sigma_acc_proc::Float64 = 0.3
    gyro_still_thresh::Float64 = 0.05 
    bias_lr::Float64 = 0.02            
    sigma_p0::Float64 = 1.0
    sigma_v0::Float64 = 0.5
    gravity::Vector{Float64} = [0.0,0.0,-9.81]
end


function run_ekf_zaru(t, acc, gyro, init_p, init_v, init_q;
                      bias_acc_fn=nothing, bias_gyro_fn=nothing,
                      config::EKFZARUConfig=EKFZARUConfig())
    n = length(t)
    x = vcat(Vector{Float64}(init_p), Vector{Float64}(init_v))
    q = QuatRotation(init_q[1], init_q[2], init_q[3], init_q[4])
    P = Matrix(Diagonal(vcat(fill(config.sigma_p0^2,3), fill(config.sigma_v0^2,3))))

    gyro_bias_est = zeros(3)

    est_pos=zeros(n,3); est_vel=zeros(n,3); est_quat=zeros(n,4)
    P_trace=zeros(n); obs_times=Int[]
    est_pos[1,:]=x[1:3]; est_vel[1,:]=x[4:6]
    est_quat[1,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]; P_trace[1]=tr(P[1:3,1:3])

    for i in 2:n
        dt = t[i]-t[i-1]; dt = dt<=0 ? 0.01 : dt

        gyro_i = Vector{Float64}(gyro[i,:]); acc_i = Vector{Float64}(acc[i,:])
        bias_gyro_fn !== nothing && (gyro_i = gyro_i .- bias_gyro_fn(i))
        bias_acc_fn  !== nothing && (acc_i  = acc_i  .- bias_acc_fn(i))

        if abs(gyro_i[3]) < config.gyro_still_thresh
            gyro_bias_est[3] = (1-config.bias_lr)*gyro_bias_est[3] + config.bias_lr*gyro_i[3]
            push!(obs_times, i)
        end
        gyro_corr = copy(gyro_i)
        gyro_corr[3] -= gyro_bias_est[3]

        dθ = gyro_corr .* dt
        q = q * QuatRotation(RotXYZ(dθ[1],dθ[2],dθ[3]))

        acc_nav = q * acc_i .+ config.gravity
        x[1:3] = x[1:3] .+ x[4:6].*dt .+ 0.5.*acc_nav.*dt^2
        x[4:6] = x[4:6] .+ acc_nav.*dt

        F = Matrix{Float64}(I,6,6); F[1:3,4:6] = I(3).*dt
        Q = zeros(6,6)
        Q[4:6,4:6] = I(3) .* (config.sigma_acc_proc^2 * dt)
        Q[1:3,1:3] = I(3) .* (0.25*config.sigma_acc_proc^2 * dt^3)
        P = F*P*F' + Q; P = 0.5*(P+P')

        est_pos[i,:]=x[1:3]; est_vel[i,:]=x[4:6]
        est_quat[i,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
        P_trace[i]=tr(P[1:3,1:3])
    end

    return est_pos, est_vel, est_quat, P_trace, obs_times
end

end 