module TrainingSupervisorV8Calibrated

using Lux
using Zygote
using Optimisers
using ComponentArrays
using Random
using LinearAlgebra
using Statistics
using Printf
using JLD2

using Main.SciML_Models
using Main.TrainingSynthCtxCalibrated

export train_supervisor_v8_calib, SupervisorV8CalibState,
       build_model_bilstm_v8_calib, integrate_bias_v8_calib,
       calibrate_drift_correction_calib, set_drift_correction_calib



struct SupervisorV8CalibState
    ml1::CtxBiasStateCalib
    sup; sup_ps; sup_st
    window_ml2::Int
    dt::Float64
    sigma_rw_acc::Float64
    sigma_rw_gyro::Float64
    in_mean::Vector{Float32}; in_std::Vector{Float32}
    out_mean::Vector{Float32}; out_std::Vector{Float32}
    slope_window::Int
    sigma_phys::Vector{Float32}
    drift_correction::Vector{Float64}
end


struct BiasBiLSTMV8Calib{F,B,H} <: Lux.AbstractLuxContainerLayer{(:fwd, :bwd, :head)}
    fwd::F
    bwd::B
    head::H
end

function build_model_bilstm_v8_calib(; hidden::Int=64, seed::Int=343,
                                       n_features::Int=6, dropout::Float64=0.3)
    fwd  = Recurrence(LSTMCell(n_features => hidden); return_sequence=false)
    bwd  = Recurrence(LSTMCell(n_features => hidden); return_sequence=false)
    head = Chain(
        Dense(2*hidden => hidden, tanh),
        Dropout(Float32(dropout)),
        Dense(hidden => 6, tanh)
    )
    model = BiasBiLSTMV8Calib(fwd, bwd, head)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

function (m::BiasBiLSTMV8Calib)(x, ps, st)
    h_fwd, st_f = m.fwd(x, ps.fwd, st.fwd)
    x_rev = reverse(x; dims=2)
    h_bwd, st_b = m.bwd(x_rev, ps.bwd, st.bwd)
    h_cat = vcat(h_fwd, h_bwd)
    y, st_h = m.head(h_cat, ps.head, st.head)
    return y, (fwd=st_f, bwd=st_b, head=st_h)
end

function build_tensors_v8_calib(runs, ml1::CtxBiasStateCalib;
                                 window_ml2::Int=10, stride::Int=8,
                                 dt::Float64=0.01,
                                 slope_window::Int=50)
    W_ML1 = ml1.window
    W_ML2 = window_ml2
    ba_calib = ml1.bias_acc_calib
    bg_calib = ml1.bias_gyro_calib
    baf, bgf = make_bias_fns_ctx_calibrated(ml1)

    Xs = Matrix{Float32}[]
    Ys = Vector{Float32}[]

    for r in runs
        n = r.n
        acc_cal  = r.acc  .- ba_calib'
        gyro_cal = r.gyro .- bg_calib'

        bias_acc_res_real  = r.bias_acc_det  .- ba_calib'
        bias_gyro_res_real = r.bias_gyro_det .- bg_calib'

        ml1_acc_total  = zeros(n, 3)
        ml1_gyro_total = zeros(n, 3)
        for i in W_ML1:n
            ml1_acc_total[i,:]  = baf(r.acc[(i-W_ML1+1):i, :])
            ml1_gyro_total[i,:] = bgf(r.gyro[(i-W_ML1+1):i, :])
        end
        ml1_acc_res  = ml1_acc_total  .- ba_calib'
        ml1_gyro_res = ml1_gyro_total .- bg_calib'

        for i in W_ML1:stride:(n - slope_window)
            win_acc_corr  = acc_cal[(i-W_ML2+1):i, :]  .- ml1_acc_res[i,:]'
            win_gyro_corr = gyro_cal[(i-W_ML2+1):i, :] .- ml1_gyro_res[i,:]'

            combined = hcat(win_acc_corr, win_gyro_corr)
            combined_t = permutedims(combined, (2,1))
            push!(Xs, Float32.(combined_t))

            interval = slope_window * dt
            d_bias_acc  = (bias_acc_res_real[i+slope_window,:]  .- bias_acc_res_real[i,:])  ./ interval
            d_bias_gyro = (bias_gyro_res_real[i+slope_window,:] .- bias_gyro_res_real[i,:]) ./ interval

            push!(Ys, Float32.(vcat(d_bias_acc, d_bias_gyro)))
        end
    end
    N = length(Xs)
    X = Array{Float32}(undef, 6, window_ml2, N)
    for j in 1:N; X[:,:,j] = Xs[j]; end
    Y = Array{Float32}(undef, 6, N)
    for j in 1:N; Y[:,j] = Ys[j]; end
    return X, Y
end

function mse_loss(pred::AbstractArray{Float32}, target::AbstractArray{Float32})
    return mean(abs2, pred .- target)
end

function nll_hinge_loss(pred::AbstractArray{Float32},
                         target::AbstractArray{Float32},
                         sigma_phys::AbstractVector{Float32};
                         λ_hinge::Float32=10.0f0)
    err_sq = (pred .- target) .^ 2

    σ_sq = reshape(sigma_phys.^2, 6, 1)
    L_NLL = mean(err_sq ./ (2.0f0 .* σ_sq))

    σ3 = reshape(3.0f0 .* sigma_phys, 6, 1)
    excess = max.(0.0f0, abs.(pred) .- σ3)
    L_hinge = mean(excess .^ 2)

    return L_NLL + λ_hinge * L_hinge
end

function train_supervisor_v8_calib(runs, ml1::CtxBiasStateCalib;
                                     epochs::Int=20, lr::Float64=3e-4,
                                     hidden::Int=64, stride::Int=8,
                                     window_ml2::Int=10,
                                     dropout::Float64=0.3,
                                     dt::Float64=0.01,
                                     sigma_rw_acc::Float64=0.005,
                                     sigma_rw_gyro::Float64=0.0003,
                                     slope_window::Int=50,
                                     checkpoint_dir::String="checkpoints_v8_calib",
                                     checkpoint_every::Int=5,
                                     resume::Bool=true,
                                     verbose::Bool=true)
    isdir(checkpoint_dir) || mkpath(checkpoint_dir)
    verbose && @printf("Supervisor V8 CALIB+SLOPE: window_ml2=%d, hidden=%d, dropout=%.2f, %d epoci\n",
                       window_ml2, hidden, dropout, epochs)
    verbose && @printf("   ML1 calibrare: acc=%s, gyro=%s\n",
                       ml1.bias_acc_calib, ml1.bias_gyro_calib)
    verbose && @printf("   slope_window=%d eșantioane (%.2fs)\n",
                       slope_window, slope_window*dt)

    verbose && println("   Pregătesc tensori (6 canale, target=pante medii reziduale)...")
    X, Y = build_tensors_v8_calib(runs, ml1; window_ml2=window_ml2,
                                  stride=stride, dt=dt, slope_window=slope_window)
    N = size(X, 3)
    verbose && @printf("   %d ferestre antrenare\n", N)
    verbose && @printf("   țintă derivată: med=%.6e std=%.6e\n", mean(Y), std(Y))

    in_mean  = Float32.(vec(mean(X, dims=(2,3))))
    in_std   = max.(Float32.(vec(std(X, dims=(2,3)))), 1f-6)

    out_mean = zeros(Float32, 6)
    out_std  = ones(Float32, 6)

    interval = slope_window * dt
    σ_phys_acc  = Float32(sigma_rw_acc  / sqrt(interval))   # m/s³ (panta acc)
    σ_phys_gyro = Float32(sigma_rw_gyro / sqrt(interval))   # rad/s² (panta gyro)
    sigma_phys = Float32[σ_phys_acc, σ_phys_acc, σ_phys_acc,
                          σ_phys_gyro, σ_phys_gyro, σ_phys_gyro]
    verbose && @printf("   σ_phys: acc=%.5e (m/s³), gyro=%.5e (rad/s²)\n",
                       σ_phys_acc, σ_phys_gyro)
    verbose && @printf("   ±3σ limita fizică: acc=%.5e, gyro=%.5e\n",
                       3*σ_phys_acc, 3*σ_phys_gyro)

    Xn = (X .- in_mean) ./ in_std
    Yn = Float32.(Y)

    model, ps, st = build_model_bilstm_v8_calib(hidden=hidden, seed=343,
                                                 n_features=6, dropout=dropout)
    ps = ComponentArray(ps)

    opt = Optimisers.Adam(lr); st_opt = Optimisers.setup(opt, ps)
    bs = 256
    best_loss = Inf
    patience_cnt = 0
    patience_max = 5
    start_epoch = 1

    if resume
        ckpt_files = filter(f -> startswith(f, "ckpt_ep") && endswith(f, ".jld2"),
                            readdir(checkpoint_dir))
        if !isempty(ckpt_files)
            ep_nums = [parse(Int, replace(f, "ckpt_ep"=>"", ".jld2"=>"")) for f in ckpt_files]
            last_ep = maximum(ep_nums)
            ckpt_path = joinpath(checkpoint_dir, "ckpt_ep$(last_ep).jld2")
            verbose && @printf("   Reiau de la checkpoint: %s\n", ckpt_path)
            d = JLD2.load(ckpt_path)
            ps         = d["ps"]
            st_opt     = d["st_opt"]
            best_loss  = d["best_loss"]
            start_epoch = last_ep + 1
        end
    end

    output_scale = reshape(3.0f0 .* sigma_phys, 6, 1)

    for ep in start_epoch:epochs
        if ep == 10
            opt = Optimisers.Adam(lr * 0.3); st_opt = Optimisers.setup(opt, ps)
        elseif ep == 15
            opt = Optimisers.Adam(lr * 0.1); st_opt = Optimisers.setup(opt, ps)
        end
        perm = randperm(N)
        ep_loss = 0.0; nb = 0
        for b0 in 1:bs:N
            idx = perm[b0:min(b0+bs-1,N)]
            xb = Xn[:,:,idx]; yb = Yn[:,idx]
            l, back = Zygote.pullback(ps) do p
                pred_raw, _ = model(xb, p, st)
                pred_phys = pred_raw .* output_scale
                nll_hinge_loss(pred_phys, yb, sigma_phys; λ_hinge=50.0f0)
            end
            g = back(1.0f0)[1]
            st_opt, ps = Optimisers.update(st_opt, ps, g)
            ep_loss += l; nb += 1
        end
        avg_loss = ep_loss/nb
        verbose && @printf("   epoca %d: loss=%.6f\n", ep, avg_loss)

        if ep % checkpoint_every == 0 || ep == epochs
            ckpt_path = joinpath(checkpoint_dir, "ckpt_ep$(ep).jld2")
            JLD2.jldsave(ckpt_path;
                ps=ps, st_opt=st_opt, best_loss=best_loss, ep=ep,
                in_mean=in_mean, in_std=in_std,
                out_mean=out_mean, out_std=out_std,
                window_ml2=window_ml2, dt=dt,
                sigma_rw_acc=sigma_rw_acc, sigma_rw_gyro=sigma_rw_gyro,
                slope_window=slope_window,
                sigma_phys=sigma_phys)
        end

        if avg_loss < best_loss - 1f-5
            best_loss = avg_loss
            patience_cnt = 0
            best_path = joinpath(checkpoint_dir, "best.jld2")
            JLD2.jldsave(best_path;
                ps=ps, best_loss=best_loss, ep=ep,
                in_mean=in_mean, in_std=in_std,
                out_mean=out_mean, out_std=out_std,
                window_ml2=window_ml2, dt=dt,
                sigma_rw_acc=sigma_rw_acc, sigma_rw_gyro=sigma_rw_gyro,
                slope_window=slope_window,
                sigma_phys=sigma_phys)
        else
            patience_cnt += 1
            if patience_cnt >= patience_max
                verbose && @printf("   Early stop la ep %d (best=%.6f)\n", ep, best_loss)
                break
            end
        end
    end

    best_path = joinpath(checkpoint_dir, "best.jld2")
    if isfile(best_path)
        d = JLD2.load(best_path)
        ps = d["ps"]
    end

    return SupervisorV8CalibState(ml1, model, ps, st, window_ml2, dt,
                                   sigma_rw_acc, sigma_rw_gyro,
                                   in_mean, in_std, out_mean, out_std,
                                   slope_window,
                                   sigma_phys,
                                   zeros(6))
end

function integrate_bias_v8_calib(r, state::SupervisorV8CalibState;
                                  residual_scale::Float64=1.0,
                                  leak::Float64=0.003,
                                  gate::Bool=true,
                                  gate_gain::Float64=1.0,
                                  leak_max::Float64=0.05)
    W_ML1 = state.ml1.window
    n = r.n
    dt = state.dt
    ba_calib = state.ml1.bias_acc_calib
    bg_calib = state.ml1.bias_gyro_calib

    acc_cal  = r.acc  .- ba_calib'
    gyro_cal = r.gyro .- bg_calib'

    ml1_baf, ml1_bgf = make_bias_fns_ctx_calibrated(state.ml1)
    ml1_acc_total  = zeros(n, 3)
    ml1_gyro_total = zeros(n, 3)
    for i in 1:n
        lo = max(1, i-W_ML1+1)
        wa = r.acc[lo:i,:]; wg = r.gyro[lo:i,:]
        if size(wa,1) < W_ML1
            pad = W_ML1 - size(wa,1)
            wa = vcat(repeat(wa[1:1,:],pad),wa); wg = vcat(repeat(wg[1:1,:],pad),wg)
        end
        ml1_acc_total[i,:]  = ml1_baf(wa)
        ml1_gyro_total[i,:] = ml1_bgf(wg)
    end

    ml1_acc_res  = ml1_acc_total  .- ba_calib'
    ml1_gyro_res = ml1_gyro_total .- bg_calib'

    bias_acc_res  = zeros(size(ml1_acc_res))
    bias_gyro_res = zeros(size(ml1_gyro_res))

    bias_acc_res[W_ML1,:]  = ml1_acc_res[W_ML1,:]
    bias_gyro_res[W_ML1,:] = ml1_gyro_res[W_ML1,:]

    W_ML2 = state.window_ml2
    M = state.slope_window
    max_d_acc  = 3.0 * state.sigma_rw_acc  / sqrt(dt)
    max_d_gyro = 3.0 * state.sigma_rw_gyro / sqrt(dt)

    α = leak

    acc_mag  = vec(sqrt.(sum(acc_cal.^2,  dims=2)))  
    gyro_mag = vec(sqrt.(sum(gyro_cal.^2, dims=2)))   
    w_gyro = mean(acc_mag) / max(mean(gyro_mag), 1e-9)
    ρ = acc_mag .+ w_gyro .* gyro_mag                
    ρ̄ = max(mean(ρ), 1e-9)
    ρ̂ = ρ ./ ρ̄                                        

    i = W_ML1 + 1
    while i <= n
        lo = max(1, i - W_ML2)
        win_acc_corr  = acc_cal[lo:i-1, :]  .- ml1_acc_res[i-1,:]'
        win_gyro_corr = gyro_cal[lo:i-1, :] .- ml1_gyro_res[i-1,:]'

        if size(win_acc_corr, 1) < W_ML2
            pad = W_ML2 - size(win_acc_corr, 1)
            win_acc_corr  = vcat(repeat(win_acc_corr[1:1,:], pad),  win_acc_corr)
            win_gyro_corr = vcat(repeat(win_gyro_corr[1:1,:], pad), win_gyro_corr)
        end

        combined = hcat(win_acc_corr, win_gyro_corr)
        combined_t = permutedims(combined, (2,1))
        x = reshape(Float32.(combined_t), 6, W_ML2, 1)
        xn = (x .- state.in_mean) ./ state.in_std
        y, _ = state.sup(xn, state.sup_ps, state.sup_st)
        y_vec = y[:, 1]

        slopes = Float64.(y_vec .* (3.0f0 .* state.sigma_phys))

        slopes .-= state.drift_correction

        s_ba = clamp.(slopes[1:3], -max_d_acc,  max_d_acc)
        s_bg = clamp.(slopes[4:6], -max_d_gyro, max_d_gyro)

        end_i = min(i + M - 1, n)
        for k in i:end_i
            if gate
                α_eff = clamp(α + gate_gain * max(0.0, ρ̂[k] - 1.0), α, leak_max)
            else
                α_eff = α
            end
            bias_acc_res[k,:]  = (1.0 - α_eff) .* bias_acc_res[k-1,:]  .+ α_eff .* ml1_acc_res[k-1,:]  .+ residual_scale .* s_ba .* dt
            bias_gyro_res[k,:] = (1.0 - α_eff) .* bias_gyro_res[k-1,:] .+ α_eff .* ml1_gyro_res[k-1,:] .+ residual_scale .* s_bg .* dt
        end

        i = end_i + 1
    end
    
    ba_total = bias_acc_res  .+ ba_calib'
    bg_total = bias_gyro_res .+ bg_calib'

    return ba_total, bg_total
end


function calibrate_drift_correction_calib(state::SupervisorV8CalibState, calibration_runs)
    W_ML1 = state.ml1.window
    W_ML2 = state.window_ml2
    ba_calib = state.ml1.bias_acc_calib
    bg_calib = state.ml1.bias_gyro_calib
    accumulator = zeros(6)
    count = 0

    original_drift = copy(state.drift_correction)

    try
        for r in calibration_runs
            n = r.n
            acc_cal  = r.acc  .- ba_calib'
            gyro_cal = r.gyro .- bg_calib'

            ml1_baf, ml1_bgf = make_bias_fns_ctx_calibrated(state.ml1)
            ml1_acc_res  = zeros(n, 3)
            ml1_gyro_res = zeros(n, 3)
            for i in W_ML1:n
                ml1_acc_res[i,:]  = ml1_baf(r.acc[(i-W_ML1+1):i, :])  .- ba_calib
                ml1_gyro_res[i,:] = ml1_bgf(r.gyro[(i-W_ML1+1):i, :]) .- bg_calib
            end

            for i in (W_ML1+1):20:n 
                lo = max(1, i - W_ML2)
                win_acc_corr  = acc_cal[lo:i-1, :]  .- ml1_acc_res[i-1,:]'
                win_gyro_corr = gyro_cal[lo:i-1, :] .- ml1_gyro_res[i-1,:]'
                if size(win_acc_corr, 1) < W_ML2
                    pad = W_ML2 - size(win_acc_corr, 1)
                    win_acc_corr  = vcat(repeat(win_acc_corr[1:1,:], pad),  win_acc_corr)
                    win_gyro_corr = vcat(repeat(win_gyro_corr[1:1,:], pad), win_gyro_corr)
                end
                combined = hcat(win_acc_corr, win_gyro_corr)
                combined_t = permutedims(combined, (2,1))
                x = reshape(Float32.(combined_t), 6, W_ML2, 1)
                xn = (x .- state.in_mean) ./ state.in_std
                y, _ = state.sup(xn, state.sup_ps, state.sup_st)
                slopes = Float64.(vec(y) .* (3.0 .* state.sigma_phys))
                accumulator .+= slopes
                count += 1
            end
        end
    finally
    end

    if count == 0
        return zeros(6)
    end
    return accumulator ./ count
end


function set_drift_correction_calib(state::SupervisorV8CalibState, drift::Vector{Float64})
    return SupervisorV8CalibState(state.ml1, state.sup, state.sup_ps, state.sup_st,
                                   state.window_ml2, state.dt,
                                   state.sigma_rw_acc, state.sigma_rw_gyro,
                                   state.in_mean, state.in_std,
                                   state.out_mean, state.out_std,
                                   state.slope_window,
                                   state.sigma_phys,
                                   drift)
end

end