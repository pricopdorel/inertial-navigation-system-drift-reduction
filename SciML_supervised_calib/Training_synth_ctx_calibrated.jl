module TrainingSynthCtxCalibrated

using Lux
using Zygote
using Optimisers
using ComponentArrays
using Random
using LinearAlgebra
using Statistics
using Printf

using Main.SciML_Models

export train_bias_ctx_calibrated, CtxBiasStateCalib,
       make_bias_fns_ctx_calibrated, build_ctx_tensors_calibrated,
       BIAS_ACC_CALIB, BIAS_GYRO_CALIB

const BIAS_ACC_CALIB  = [0.1, -0.08, 0.06]      # m/s²
const BIAS_GYRO_CALIB = [0.01, -0.008, 0.006]   # rad/s

struct CtxBiasStateCalib
    model_A; ps_A; st_A
    model_G; ps_G; st_G
    window::Int
    bias_acc_calib::Vector{Float64}
    bias_gyro_calib::Vector{Float64}
    in_mean_A::Vector{Float32}; in_std_A::Vector{Float32}
    out_mean_A::Vector{Float32}; out_std_A::Vector{Float32}
    in_mean_G::Vector{Float32}; in_std_G::Vector{Float32}
    out_mean_G::Vector{Float32}; out_std_G::Vector{Float32}
end

function build_ctx_tensors_calibrated(runs, window, sensor::Symbol;
                                       stride::Int=1,
                                       bias_acc_calib::Vector{Float64}=BIAS_ACC_CALIB,
                                       bias_gyro_calib::Vector{Float64}=BIAS_GYRO_CALIB)
    Xs = Matrix{Float32}[]
    Ys = Vector{Float32}[]
    bias_calib = sensor == :acc ? bias_acc_calib : bias_gyro_calib

    for r in runs
        sig_raw  = sensor == :acc ? r.acc : r.gyro
        sig_calib = sig_raw .- bias_calib'

        bias_total_det = sensor == :acc ? r.bias_acc_det : r.bias_gyro_det
        bias_residual  = bias_total_det .- bias_calib'

        for i in window:stride:r.n
            win = permutedims(sig_calib[(i-window+1):i, :], (2,1))   # 3×W
            push!(Xs, Float32.(win))
            push!(Ys, Float32.(bias_residual[i, :]))
        end
    end
    N = length(Xs)
    X = Array{Float32}(undef, 3, window, N)
    for j in 1:N; X[:,:,j] = Xs[j]; end
    Y = reduce(hcat, Ys)
    return X, Y
end

function train_bias_ctx_calibrated(runs;
                                    window::Int=30,
                                    epochs::Int=30,
                                    lr::Float64=1e-3,
                                    hidden::Int=128,
                                    stride::Int=5,
                                    bias_acc_calib::Vector{Float64}=BIAS_ACC_CALIB,
                                    bias_gyro_calib::Vector{Float64}=BIAS_GYRO_CALIB,
                                    verbose::Bool=true)
    verbose && println("="^60)
    verbose && println("Antrenare ML1 CALIBRATED (cu bias offset removal)")
    verbose && @printf("Calibrare acc:  %s\n", bias_acc_calib)
    verbose && @printf("Calibrare gyro: %s\n", bias_gyro_calib)
    verbose && println("="^60)

    verbose && println("\n[A] Construire tensori ACC calibrați...")
    XA, YA = build_ctx_tensors_calibrated(runs, window, :acc;
                                            stride=stride,
                                            bias_acc_calib=bias_acc_calib,
                                            bias_gyro_calib=bias_gyro_calib)
    verbose && @printf("    %d ferestre, țintă: med=%.5f std=%.5f\n",
                       size(XA,3), mean(YA), std(YA))

    in_mean_A  = Float32.(vec(mean(XA, dims=(2,3))))
    in_std_A   = max.(Float32.(vec(std(XA, dims=(2,3)))), 1f-6)
    out_mean_A = Float32.(vec(mean(YA, dims=2)))
    out_std_A  = max.(Float32.(vec(std(YA, dims=2))), 1f-6)
    XnA = (XA .- in_mean_A) ./ in_std_A
    YnA = (YA .- out_mean_A) ./ out_std_A

    model_A, ps_A, st_A = SciML_Models.build_model_lstm_ctx(hidden=hidden, seed=42)
    ps_A = ComponentArray(ps_A)
    opt = Optimisers.Adam(lr); st_opt = Optimisers.setup(opt, ps_A)
    bs = 256
    NA = size(XA, 3)
    for ep in 1:epochs
        if ep == epochs÷3 || ep == 2*epochs÷3
            lr2 = lr * (ep==epochs÷3 ? 0.3 : 0.1)
            opt = Optimisers.Adam(lr2); st_opt = Optimisers.setup(opt, ps_A)
        end
        perm = randperm(NA)
        ep_loss = 0.0; nb = 0
        for b0 in 1:bs:NA
            idx = perm[b0:min(b0+bs-1, NA)]
            xb = XnA[:,:,idx]; yb = YnA[:,idx]
            l, back = Zygote.pullback(ps_A) do p
                pred, _ = model_A(xb, p, st_A)
                mean(abs2, pred .- yb)
            end
            g = back(1.0f0)[1]
            st_opt, ps_A = Optimisers.update(st_opt, ps_A, g)
            ep_loss += l; nb += 1
        end
        verbose && @printf("    acc epoca %d: loss=%.6f\n", ep, ep_loss/nb)
    end

    verbose && println("\n[G] Construire tensori GYRO calibrați...")
    XG, YG = build_ctx_tensors_calibrated(runs, window, :gyro;
                                            stride=stride,
                                            bias_acc_calib=bias_acc_calib,
                                            bias_gyro_calib=bias_gyro_calib)
    verbose && @printf("    %d ferestre, țintă: med=%.5f std=%.5f\n",
                       size(XG,3), mean(YG), std(YG))

    in_mean_G  = Float32.(vec(mean(XG, dims=(2,3))))
    in_std_G   = max.(Float32.(vec(std(XG, dims=(2,3)))), 1f-6)
    out_mean_G = Float32.(vec(mean(YG, dims=2)))
    out_std_G  = max.(Float32.(vec(std(YG, dims=2))), 1f-6)
    XnG = (XG .- in_mean_G) ./ in_std_G
    YnG = (YG .- out_mean_G) ./ out_std_G

    model_G, ps_G, st_G = SciML_Models.build_model_lstm_ctx(hidden=hidden, seed=43)
    ps_G = ComponentArray(ps_G)
    opt = Optimisers.Adam(lr); st_opt = Optimisers.setup(opt, ps_G)
    NG = size(XG, 3)
    for ep in 1:epochs
        if ep == epochs÷3 || ep == 2*epochs÷3
            lr2 = lr * (ep==epochs÷3 ? 0.3 : 0.1)
            opt = Optimisers.Adam(lr2); st_opt = Optimisers.setup(opt, ps_G)
        end
        perm = randperm(NG)
        ep_loss = 0.0; nb = 0
        for b0 in 1:bs:NG
            idx = perm[b0:min(b0+bs-1, NG)]
            xb = XnG[:,:,idx]; yb = YnG[:,idx]
            l, back = Zygote.pullback(ps_G) do p
                pred, _ = model_G(xb, p, st_G)
                mean(abs2, pred .- yb)
            end
            g = back(1.0f0)[1]
            st_opt, ps_G = Optimisers.update(st_opt, ps_G, g)
            ep_loss += l; nb += 1
        end
        verbose && @printf("    gyro epoca %d: loss=%.6f\n", ep, ep_loss/nb)
    end

    return CtxBiasStateCalib(model_A, ps_A, st_A,
                             model_G, ps_G, st_G,
                             window,
                             bias_acc_calib, bias_gyro_calib,
                             in_mean_A, in_std_A, out_mean_A, out_std_A,
                             in_mean_G, in_std_G, out_mean_G, out_std_G)
end

function make_bias_fns_ctx_calibrated(state::CtxBiasStateCalib)
    ba_calib = state.bias_acc_calib
    bg_calib = state.bias_gyro_calib

    function bias_acc_win(acc_win::AbstractMatrix)
        acc_calibrated = acc_win .- ba_calib'
        x = reshape(Float32.(permutedims(acc_calibrated, (2,1))), 3, size(acc_calibrated,1), 1)
        xn = (x .- state.in_mean_A) ./ state.in_std_A
        y, _ = state.model_A(xn, state.ps_A, state.st_A)
        y_vec = y[:, 1]
        residual = Float64.(y_vec .* state.out_std_A .+ state.out_mean_A)
        return ba_calib .+ residual
    end

    function bias_gyro_win(gyro_win::AbstractMatrix)
        gyro_calibrated = gyro_win .- bg_calib'
        x = reshape(Float32.(permutedims(gyro_calibrated, (2,1))), 3, size(gyro_calibrated,1), 1)
        xn = (x .- state.in_mean_G) ./ state.in_std_G
        y, _ = state.model_G(xn, state.ps_G, state.st_G)
        y_vec = y[:, 1]
        residual = Float64.(y_vec .* state.out_std_G .+ state.out_mean_G)
        return bg_calib .+ residual
    end

    return bias_acc_win, bias_gyro_win
end

end
