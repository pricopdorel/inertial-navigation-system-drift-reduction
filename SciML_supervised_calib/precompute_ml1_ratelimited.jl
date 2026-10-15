

function precompute_ml1_calib_bias_limited(r, ml1;
        rw_acc::Float64=0.005, rw_gyro::Float64=0.0003,
        k_tol::Float64=4.0,         
        ema_alpha::Float64=0.0)
    baf, bgf = make_bias_fns_ctx_calibrated(ml1)
    W = ml1.window; n = r.n
    dt = length(r.t) > 1 ? (r.t[2]-r.t[1]) : 0.01

    max_step_a = k_tol * rw_acc  * sqrt(dt)
    max_step_g = k_tol * rw_gyro * sqrt(dt)

    ba = zeros(n,3); bg = zeros(n,3)

    ba_prev = copy(ml1.bias_acc_calib)
    bg_prev = copy(ml1.bias_gyro_calib)

    for i in 1:n
        lo = max(1, i-W+1)
        wa = r.acc[lo:i,:]; wg = r.gyro[lo:i,:]
        if size(wa,1) < W
            pad = W - size(wa,1)
            wa = vcat(repeat(wa[1:1,:],pad),wa); wg = vcat(repeat(wg[1:1,:],pad),wg)
        end
        ba_raw = baf(wa); bg_raw = bgf(wg)

        for ax in 1:3
            da = ba_raw[ax] - ba_prev[ax]
            da = clamp(da, -max_step_a, max_step_a)
            ba[i,ax] = ba_prev[ax] + da

            dg = bg_raw[ax] - bg_prev[ax]
            dg = clamp(dg, -max_step_g, max_step_g)
            bg[i,ax] = bg_prev[ax] + dg
        end

        if ema_alpha > 0
            ba[i,:] = ema_alpha .* ba_prev .+ (1-ema_alpha) .* ba[i,:]
            bg[i,:] = ema_alpha .* bg_prev .+ (1-ema_alpha) .* bg[i,:]
        end

        ba_prev = ba[i,:]; bg_prev = bg[i,:]
    end
    return ba, bg
end
