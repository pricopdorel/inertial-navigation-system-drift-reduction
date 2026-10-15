    module Mechanization

    using DataFrames
    using Rotations
    using LinearAlgebra

    export run_mechanization, mechanize_arrays

    function mechanize_arrays(t::Vector{Float64},
                            gyro::AbstractMatrix,
                            acc::AbstractMatrix,
                            q0::QuatRotation,
                            p0::Vector{Float64},
                            v0::Vector{Float64};
                            bias_acc_fn=nothing,
                            bias_gyro_fn=nothing,
                            subtract_gravity::Bool=false,
                            gravity::Vector{Float64}=[0.0, 0.0, -9.81])

        n = length(t)
        est_pos  = zeros(n, 3)
        est_vel  = zeros(n, 3)
        est_quat = zeros(n, 4)

        current_q   = q0
        current_pos = copy(p0)
        current_vel = copy(v0)

        est_pos[1, :]  = current_pos
        est_vel[1, :]  = current_vel
        est_quat[1, :] = [current_q.q.v1, current_q.q.v2, current_q.q.v3, current_q.q.s]

        for i in 2:n
            dt = t[i] - t[i-1]
            if dt <= 0.0
                dt = 0.01
            end

            gyro_i = Vector{Float64}(gyro[i, :])
            acc_i  = Vector{Float64}(acc[i, :])

            if bias_gyro_fn !== nothing
                gyro_i = gyro_i .- bias_gyro_fn(gyro_i)
            end
            if bias_acc_fn !== nothing
                acc_i = acc_i .- bias_acc_fn(acc_i)
            end

            delta_rot = QuatRotation(RotXYZ(gyro_i[1]*dt, gyro_i[2]*dt, gyro_i[3]*dt))
            current_q = current_q * delta_rot

            acc_nav = current_q * acc_i

            if subtract_gravity
                acc_nav = acc_nav .+ gravity
            end

            current_vel = current_vel .+ acc_nav .* dt
            current_pos = current_pos .+ current_vel .* dt

            est_pos[i, :]  = current_pos
            est_vel[i, :]  = current_vel
            est_quat[i, :] = [current_q.q.v1, current_q.q.v2, current_q.q.v3, current_q.q.s]
        end

        return est_pos, est_vel, est_quat
    end

    function run_mechanization(imu_data; init_position, init_velocity=zeros(3), init_quaternion, kwargs...)
        n = nrow(imu_data)
        t = Vector{Float64}(imu_data.time)
        gyro = hcat(imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z)
        acc  = hcat(imu_data.acc_x,  imu_data.acc_y,  imu_data.acc_z)

        q0 = QuatRotation(
            init_quaternion[1],
            init_quaternion[2],
            init_quaternion[3],
            init_quaternion[4]
        )

        p0 = Vector{Float64}(init_position)
        v0 = Vector{Float64}(init_velocity)

        est_pos, est_vel, est_quat = mechanize_arrays(t, gyro, acc, q0, p0, v0; kwargs...)

        return DataFrame(
            time = t,
            est_x = est_pos[:, 1], est_y = est_pos[:, 2], est_z = est_pos[:, 3],
            est_vx = est_vel[:, 1], est_vy = est_vel[:, 2], est_vz = est_vel[:, 3],
            est_qx = est_quat[:, 1], est_qy = est_quat[:, 2], est_qz = est_quat[:, 3], est_qw = est_quat[:, 4],
        )
    end

    end