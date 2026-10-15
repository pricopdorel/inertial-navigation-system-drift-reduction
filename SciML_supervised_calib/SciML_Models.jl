module SciML_Models

using Lux
using Random
using LinearAlgebra
using ComponentArrays

export build_model_A, build_model_G, ErrorModel,
       preprocess, postprocess, NormStats, fit_norm_stats,
       build_model_A_ctx, build_model_G_ctx,
       build_model_lstm_ctx,
       build_model_bilstm_ctx, BiasBiLSTM

function build_model_A(; hidden=(64, 32), seed::Int=42)
    return _build_error_net(hidden, seed)
end

function build_model_G(; hidden=(64, 32), seed::Int=43)
    return _build_error_net(hidden, seed)
end

function _build_error_net(hidden::Tuple, seed::Int)
    layers = []
    in_dim = 3
    for h in hidden
        push!(layers, Dense(in_dim => h, tanh))
        in_dim = h
    end
    push!(layers, Dense(in_dim => 3))

    model = Chain(layers...)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

build_model_A_ctx(; window::Int=50, hidden=(128,64), seed::Int=42) =
    _build_ctx_net(window, hidden, seed)

build_model_G_ctx(; window::Int=50, hidden=(128,64), seed::Int=43) =
    _build_ctx_net(window, hidden, seed)

function _build_ctx_net(window::Int, hidden::Tuple, seed::Int)
    layers = []
    in_dim = 3 * window
    for h in hidden
        push!(layers, Dense(in_dim => h, tanh))
        in_dim = h
    end
    push!(layers, Dense(in_dim => 3))
    model = Chain(layers...)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

struct BiasLSTM{R,D} <: Lux.AbstractLuxContainerLayer{(:rnn, :head)}
    rnn::R
    head::D
end

function build_model_lstm_ctx(; hidden::Int=128, seed::Int=42, n_features::Int=3)
    rnn  = Recurrence(LSTMCell(n_features => hidden); return_sequence=false)
    head = Chain(Dense(hidden => hidden, tanh), Dense(hidden => 3))
    model = BiasLSTM(rnn, head)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end


struct BiasBiLSTM{F,B,H} <: Lux.AbstractLuxContainerLayer{(:fwd, :bwd, :head)}
    fwd::F
    bwd::B
    head::H
end


function build_model_bilstm_ctx(; hidden::Int=128, seed::Int=42, n_features::Int=9)
    fwd  = Recurrence(LSTMCell(n_features => hidden); return_sequence=false)
    bwd  = Recurrence(LSTMCell(n_features => hidden); return_sequence=false)
    head = Chain(Dense(2*hidden => hidden, tanh),
                 Dense(hidden => hidden, tanh),
                 Dense(hidden => 3))
    model = BiasBiLSTM(fwd, bwd, head)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    ps, st = Lux.setup(rng, model)
    return model, ps, st
end

function (m::BiasBiLSTM)(x, ps, st)
    h_fwd, st_f = m.fwd(x, ps.fwd, st.fwd)
    x_rev = reverse(x; dims=2)
    h_bwd, st_b = m.bwd(x_rev, ps.bwd, st.bwd)
    h_cat = vcat(h_fwd, h_bwd)
    y, st_h = m.head(h_cat, ps.head, st.head)
    return y, (fwd=st_f, bwd=st_b, head=st_h)
end

function (m::BiasLSTM)(x, ps, st)
    # x: (3, window, batch)
    h, st_r = m.rnn(x, ps.rnn, st.rnn)        # (hidden, batch)
    y, st_h = m.head(h, ps.head, st.head)     # (3, batch)
    return y, (rnn=st_r, head=st_h)
end


struct NormStats
    mean::Vector{Float64}
    std::Vector{Float64}
end


function fit_norm_stats(data::AbstractMatrix)
    μ = vec(sum(data, dims=1) ./ size(data, 1))
    σ = vec(sqrt.(sum((data .- μ').^2, dims=1) ./ size(data, 1)))
    σ = max.(σ, 1e-8) 
    return NormStats(μ, σ)
end


preprocess(x::AbstractVector, stats::NormStats) = (x .- stats.mean) ./ stats.std

postprocess(y::AbstractVector, stats::NormStats) = y .* stats.std



mutable struct ErrorModel
    model        
    ps           
    st           
    in_stats::NormStats   
    out_stats::NormStats  
end


function (em::ErrorModel)(x::AbstractVector)
    xn = preprocess(x, em.in_stats)
    yn, _ = Lux.apply(em.model, xn, em.ps, em.st)
    return postprocess(yn, em.out_stats)
end

end 