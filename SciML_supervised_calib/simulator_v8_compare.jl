using Pkg
Pkg.activate(".")


include("SyntheticData.jl")
include("SciML_Models.jl")
include("Mechanization.jl")
include("EKF_adaptive.jl")
include("EKF_classic.jl")
include("Training_synth_ctx_calibrated.jl")
include("Training_supervisor_v8_calibrated.jl")

using .SyntheticData
using .SciML_Models
using .Mechanization
using .EKFAdaptive
using .EKFClassic
using .TrainingSynthCtxCalibrated
using .TrainingSupervisorV8Calibrated
using LinearAlgebra
using Statistics
using Printf
using JLD2
using Rotations
using GLMakie

const ML1_FILE = "synth_ctx_models_calib.jld2"
const ML2_FILE = "synth_ctx_models_supervisor_v8_calib.jld2"

println("="^70)
println("  SIMULATOR INS — COMPARAȚIE 2 METODE — stiva V8 calibrată")
println("="^70)

isfile(ML1_FILE) || error("$ML1_FILE nu există. Rulează main_synth_calibrated.jl întâi.")
d_ml1 = JLD2.load(ML1_FILE); ML1 = d_ml1["state"]
println("ML1 calibrat încărcat (window=$(ML1.window))")

HAS_ML2 = isfile(ML2_FILE)
SUP_V8 = if HAS_ML2
    d = JLD2.load(ML2_FILE); println("ML2 V8 încărcat"); d["supervisor_v8"]
else
    println("[!] $ML2_FILE lipsă — ML1+V8 va folosi doar ML1"); nothing
end

const BG_COLOR       = RGBf(0.08,0.08,0.10);  const PANEL_COLOR = RGBf(0.14,0.14,0.18)
const AXIS_COLOR     = RGBf(0.20,0.20,0.25);  const TEXT_COLOR  = RGBf(0.92,0.92,0.95)
const ACCENT_COLOR   = RGBf(0.35,0.75,1.0);   const GT_COLOR    = RGBf(0.70,0.70,0.78)
const EST_COLOR      = RGBf(1.0,0.45,0.30)    # metoda 1 (portocaliu)
const EST2_COLOR     = RGBf(0.30,0.90,0.65)   # metoda 2 (verde-cyan)
const HEAD_GT_COLOR  = RGBf(1.0,1.0,1.0)
const HEAD_EST_COLOR = RGBf(1.0,0.35,0.25);   const HEAD_EST2_COLOR = RGBf(0.25,1.0,0.70)
const ERR_COLOR      = RGBf(1.0,0.85,0.0)
const BTN_START      = RGBf(0.20,0.65,0.40);  const BTN_PAUSE = RGBf(0.85,0.65,0.20)
const BTN_RESET      = RGBf(0.85,0.30,0.30)
const MENU_BG        = RGBf(0.22,0.22,0.28);  const MENU_BG_HL = RGBf(0.30,0.40,0.55)
const MENU_TXT       = RGBf(0.97,0.97,1.0)

const CORR_SPEC = CorruptionSpec(
    bias_acc=BIAS_ACC_CALIB, bias_gyro=BIAS_GYRO_CALIB,
    noise_acc=0.03, noise_gyro=0.0015, rw_acc=0.005, rw_gyro=0.0003, seed=103)

generate_run(shape::Symbol) = SyntheticData.generate_dataset(
    [TrajectorySpec(shape=shape, duration=25.0, scale=150.0, laps=4, seed=3)],
    CORR_SPEC; verbose=false)[1]

function precompute_ml1_bias(r, ml1)
    baf, bgf = make_bias_fns_ctx_calibrated(ml1)
    W = ml1.window; n = r.n; ba=zeros(n,3); bg=zeros(n,3)
    for i in 1:n
        lo=max(1,i-W+1); wa=r.acc[lo:i,:]; wg=r.gyro[lo:i,:]
        if size(wa,1)<W
            pad=W-size(wa,1); wa=vcat(repeat(wa[1:1,:],pad),wa); wg=vcat(repeat(wg[1:1,:],pad),wg)
        end
        ba[i,:]=baf(wa); bg[i,:]=bgf(wg)
    end
    return ba,bg
end
precompute_ml1_v8_bias(r,ml1,sup) = sup===nothing ? precompute_ml1_bias(r,ml1) :
    integrate_bias_v8_calib(r,sup; residual_scale=1.0, gate=false)

function mechanize_full(r, ba_pred, bg_pred)
    n=r.n; g_world=[0.0,0.0,-9.81]; pos=zeros(n,3); vel=zeros(n,3); quat=zeros(n,4)
    q=QuatRotation(r.quat[1,1],r.quat[1,2],r.quat[1,3],r.quat[1,4])
    pos[1,:]=r.pos[1,:]; vel[1,:]=r.vel[1,:]; quat[1,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
    for i in 2:n
        dt=r.t[i]-r.t[i-1]; dt=dt<=0 ? 0.01 : dt
        gyro_i=r.gyro[i,:].-bg_pred[i,:]; acc_i=r.acc[i,:].-ba_pred[i,:]
        dθ=gyro_i.*dt; q=q*QuatRotation(RotXYZ(dθ[1],dθ[2],dθ[3]))
        acc_nav=q*acc_i.+g_world
        vel[i,:]=vel[i-1,:].+acc_nav.*dt
        pos[i,:]=pos[i-1,:].+vel[i-1,:].*dt.+0.5.*acc_nav.*dt^2
        quat[i,:]=[q.q.s,q.q.v1,q.q.v2,q.q.v3]
    end
    return (pos=pos,vel=vel,quat=quat)
end

function trajectory_ekf_smooth(r, ml1)
    ba,bg=precompute_ml1_bias(r,ml1); baf=i->ba[i,:]; bgf=i->bg[i,:]
    cfg=EKFConfig(period_max=10_000_000, use_adaptive=false, use_periodic=false, obs_velocity=false)
    epos,evel,equat,_,_=run_ekf_adaptive(r.t,r.acc,r.gyro,r.pos[1,:],r.vel[1,:],r.quat[1,:],r.pos;
        gt_vel=nothing, bias_acc_fn=baf, bias_gyro_fn=bgf, config=cfg)
    return (pos=epos,vel=evel,quat=equat)
end

_mkfn(B) = (i -> B[i,:])

function ekf_nhc_traj(r, ba, bg)
    epos,evel,equat,_,_ = run_ekf_nhc(r.t,r.acc,r.gyro,r.pos[1,:],r.vel[1,:],r.quat[1,:];
        bias_acc_fn=_mkfn(ba), bias_gyro_fn=_mkfn(bg), config=EKFNHCConfig())
    return (pos=epos,vel=evel,quat=equat)
end
function ekf_zaru_traj(r, ba, bg)
    epos,evel,equat,_,_ = run_ekf_zaru(r.t,r.acc,r.gyro,r.pos[1,:],r.vel[1,:],r.quat[1,:];
        bias_acc_fn=_mkfn(ba), bias_gyro_fn=_mkfn(bg), config=EKFZARUConfig())
    return (pos=epos,vel=evel,quat=equat)
end

function _bias_for(r, biasmode::Symbol)
    if biasmode==:mech;  return zeros(r.n,3), zeros(r.n,3)
    elseif biasmode==:ml1; return precompute_ml1_bias(r,ML1)
    elseif biasmode==:ml1_v8; return precompute_ml1_v8_bias(r,ML1,SUP_V8)
    else error("bias necunoscut: $biasmode") end
end

function compute_state(r, method::Symbol)
    method==:gt && return (pos=r.pos,vel=r.vel,quat=r.quat), zeros(r.n,3), zeros(r.n,3)

    biasmode, ekfmode =
        method==:mech_none    ? (:mech,   :none) :
        method==:mech_nhc     ? (:mech,   :nhc)  :
        method==:mech_zaru    ? (:mech,   :zaru) :
        method==:ml1_none     ? (:ml1,    :none) :
        method==:ml1_nhc      ? (:ml1,    :nhc)  :
        method==:ml1_zaru     ? (:ml1,    :zaru) :
        method==:ml1v8_none   ? (:ml1_v8, :none) :
        method==:ml1v8_nhc    ? (:ml1_v8, :nhc)  :
        method==:ml1v8_zaru   ? (:ml1_v8, :zaru) :
        error("Metodă necunoscută: $method")

    ba, bg = _bias_for(r, biasmode)
    state =
        ekfmode==:none ? mechanize_full(r, ba, bg) :
        ekfmode==:nhc  ? ekf_nhc_traj(r, ba, bg)   :
                         ekf_zaru_traj(r, ba, bg)
    return state, ba, bg
end

quat_to_euler(qr)=(e=RotXYZ(QuatRotation(qr[1],qr[2],qr[3],qr[4])); (e.theta1,e.theta2,e.theta3))
function euler_matrix(quat)
    n=size(quat,1); E=zeros(n,3)
    for i in 1:n; r,p,y=quat_to_euler(quat[i,:]); E[i,:]=[r,p,y]; end
    return E
end

function launch_simulator()
    GLMakie.activate!()
    set_theme!(Theme(backgroundcolor=BG_COLOR, textcolor=TEXT_COLOR,
        Axis3=(backgroundcolor=BG_COLOR, xgridcolor=AXIS_COLOR, ygridcolor=AXIS_COLOR,
               zgridcolor=AXIS_COLOR, xtickcolor=TEXT_COLOR, ytickcolor=TEXT_COLOR,
               ztickcolor=TEXT_COLOR, xlabelcolor=TEXT_COLOR, ylabelcolor=TEXT_COLOR,
               zlabelcolor=TEXT_COLOR, xticklabelcolor=TEXT_COLOR, yticklabelcolor=TEXT_COLOR,
               zticklabelcolor=TEXT_COLOR, titlecolor=TEXT_COLOR),
        Axis=(backgroundcolor=PANEL_COLOR, xgridcolor=AXIS_COLOR, ygridcolor=AXIS_COLOR,
              xtickcolor=TEXT_COLOR, ytickcolor=TEXT_COLOR, xlabelcolor=TEXT_COLOR,
              ylabelcolor=TEXT_COLOR, xticklabelcolor=TEXT_COLOR, yticklabelcolor=TEXT_COLOR,
              titlecolor=TEXT_COLOR, leftspinecolor=AXIS_COLOR, rightspinecolor=AXIS_COLOR,
              topspinecolor=AXIS_COLOR, bottomspinecolor=AXIS_COLOR)))

    method_options=[
        "GT (referință)",
        "Mec — fără EKF",        "Mec — EKF-ZARU",        "Mec — EKF-NHC",
        "ML bias — fără EKF",    "ML bias — EKF-ZARU",    "ML bias — EKF-NHC",
        "ML bias+sup — fără EKF","ML bias+sup — EKF-ZARU","ML bias+sup — EKF-NHC",
    ]
    method_symbols=[
        :gt,
        :mech_none,  :mech_zaru,  :mech_nhc,
        :ml1_none,   :ml1_zaru,   :ml1_nhc,
        :ml1v8_none, :ml1v8_zaru, :ml1v8_nhc,
    ]
    shape_options=["circle","figure8","rectangle","spiral"]
    shape_symbols=[:circle,:figure8,:rectangle,:spiral]
    stat_options=["Bias acc","Bias gyro","Eroare poziție","Eroare orientare","Eroare viteză"]
    stat_keys=[:bias_acc,:bias_gyro,:err_pos,:err_ori,:err_vel]
    comp_options=["X","Y","Z","Total"]

    frame_idx=Observable(1); is_playing=Observable(false)
    run_data=Observable(generate_run(:circle))

    s1,ba1,bg1 = compute_state(run_data[], :ml1_none)
    s2,ba2_,bg2_ = compute_state(run_data[], :mech_none)
    est_state =Observable(s1);  est_ba =Observable(ba1);  est_bg =Observable(bg1)
    est_state2=Observable(s2);  est_ba2=Observable(ba2_); est_bg2=Observable(bg2_)

    gt_traj=@lift($run_data.pos)

    fig=Figure(size=(1560,900), backgroundcolor=BG_COLOR)
    ax3d=Axis3(fig[1:2,1:3], title="Simulator INS — comparație 2 metode", xlabel="X (m)",
           ylabel="Y (m)", zlabel="Z (m)", aspect=(1,1,1), azimuth=π/4, elevation=π/8,
           titlesize=22, protrusions=30)

    lines!(ax3d, @lift(Point3f.($gt_traj[:,1],$gt_traj[:,2],$gt_traj[:,3])), color=(GT_COLOR,0.30), linewidth=2)

    est_pos_obs =@lift($est_state.pos);  est_pos_obs2=@lift($est_state2.pos)
    trail_est=@lift begin n=min($frame_idx,size($est_pos_obs,1))
        Point3f.($est_pos_obs[1:n,1],$est_pos_obs[1:n,2],$est_pos_obs[1:n,3]) end
    trail_est2=@lift begin n=min($frame_idx,size($est_pos_obs2,1))
        Point3f.($est_pos_obs2[1:n,1],$est_pos_obs2[1:n,2],$est_pos_obs2[1:n,3]) end
    lines!(ax3d, trail_est,  color=EST_COLOR,  linewidth=3.5)
    lines!(ax3d, trail_est2, color=EST2_COLOR, linewidth=3.5)
    trail_gt=@lift begin n=min($frame_idx,size($gt_traj,1))
        Point3f.($gt_traj[1:n,1],$gt_traj[1:n,2],$gt_traj[1:n,3]) end
    lines!(ax3d, trail_gt, color=GT_COLOR, linewidth=2.5, linestyle=:dash)

    head_est =@lift begin i=min($frame_idx,size($est_pos_obs,1)); Point3f($est_pos_obs[i,1],$est_pos_obs[i,2],$est_pos_obs[i,3]) end
    head_est2=@lift begin i=min($frame_idx,size($est_pos_obs2,1)); Point3f($est_pos_obs2[i,1],$est_pos_obs2[i,2],$est_pos_obs2[i,3]) end
    head_gt  =@lift begin i=min($frame_idx,size($gt_traj,1)); Point3f($gt_traj[i,1],$gt_traj[i,2],$gt_traj[i,3]) end
    scatter!(ax3d, @lift([$head_gt]),  color=HEAD_GT_COLOR,  markersize=18, strokecolor=AXIS_COLOR, strokewidth=2)
    scatter!(ax3d, @lift([$head_est]), color=HEAD_EST_COLOR, markersize=24, strokecolor=:white, strokewidth=2)
    scatter!(ax3d, @lift([$head_est2]),color=HEAD_EST2_COLOR,markersize=24, strokecolor=:white, strokewidth=2)

    stat_ax=Axis(fig[3,1:3], xlabel="Timp (s)", ylabel="", title="Statistici", titlesize=16, height=200)

    panel=fig[1:3,4]=GridLayout(); rowg=Ref(0); nextrow()=(rowg[]+=1; rowg[])
    Label(panel[nextrow(),1],"── CONTROL ──", fontsize=18, font=:bold, halign=:center, color=ACCENT_COLOR)
    Label(panel[nextrow(),1],"Traseu:", fontsize=13, halign=:left, color=TEXT_COLOR)
    shape_menu=Menu(panel[nextrow(),1], options=shape_options, default="circle", width=210,
        cell_color_active=MENU_BG_HL, cell_color_hover=MENU_BG_HL, cell_color_inactive_even=MENU_BG,
        cell_color_inactive_odd=MENU_BG, textcolor=MENU_TXT, selection_cell_color_inactive=MENU_BG)
    Label(panel[nextrow(),1],"Metoda 1 (portocaliu):", fontsize=13, halign=:left, color=EST_COLOR)
    method_menu=Menu(panel[nextrow(),1], options=method_options, default="ML bias — fără EKF", width=210,
        cell_color_active=MENU_BG_HL, cell_color_hover=MENU_BG_HL, cell_color_inactive_even=MENU_BG,
        cell_color_inactive_odd=MENU_BG, textcolor=MENU_TXT, selection_cell_color_inactive=MENU_BG)
    Label(panel[nextrow(),1],"Metoda 2 (verde):", fontsize=13, halign=:left, color=EST2_COLOR)
    method_menu2=Menu(panel[nextrow(),1], options=method_options, default="Mec — fără EKF", width=210,
        cell_color_active=MENU_BG_HL, cell_color_hover=MENU_BG_HL, cell_color_inactive_even=MENU_BG,
        cell_color_inactive_odd=MENU_BG, textcolor=MENU_TXT, selection_cell_color_inactive=MENU_BG)

    btn_grid=panel[nextrow(),1]=GridLayout()
    btn_start=Button(btn_grid[1,1], label="START", width=66, height=36, buttoncolor=BTN_START, labelcolor=:white, fontsize=13)
    btn_pause=Button(btn_grid[1,2], label="PAUZĂ", width=66, height=36, buttoncolor=BTN_PAUSE, labelcolor=:black, fontsize=13)
    btn_reset=Button(btn_grid[1,3], label="RESET", width=66, height=36, buttoncolor=BTN_RESET, labelcolor=:white, fontsize=13)

    Label(panel[nextrow(),1],"Viteză animație:", fontsize=13, color=TEXT_COLOR)
    speed_slider=Slider(panel[nextrow(),1], range=1:1:50, startvalue=10, width=210)
    Label(panel[nextrow(),1], @lift("$(round(Int,$(speed_slider.value))) pași/frame"), fontsize=11, color=TEXT_COLOR)

    Label(panel[nextrow(),1],"── STATISTICI ──", fontsize=15, font=:bold, halign=:center, color=ACCENT_COLOR)
    Label(panel[nextrow(),1],"Mărime:", fontsize=13, halign=:left, color=TEXT_COLOR)
    stat_menu=Menu(panel[nextrow(),1], options=stat_options, default="Eroare poziție", width=210,
        cell_color_active=MENU_BG_HL, cell_color_hover=MENU_BG_HL, cell_color_inactive_even=MENU_BG,
        cell_color_inactive_odd=MENU_BG, textcolor=MENU_TXT, selection_cell_color_inactive=MENU_BG)
    Label(panel[nextrow(),1],"Componentă:", fontsize=13, halign=:left, color=TEXT_COLOR)
    comp_menu=Menu(panel[nextrow(),1], options=comp_options, default="Total", width=210,
        cell_color_active=MENU_BG_HL, cell_color_hover=MENU_BG_HL, cell_color_inactive_even=MENU_BG,
        cell_color_inactive_odd=MENU_BG, textcolor=MENU_TXT, selection_cell_color_inactive=MENU_BG)

    Label(panel[nextrow(),1],"── INFO LIVE ──", fontsize=14, font=:bold, halign=:center, color=ACCENT_COLOR)
    time_label=Observable("Timp: 0.00 s")
    rmse_label =Observable("RMSE metoda 1: 0.00 m")
    rmse_label2=Observable("RMSE metoda 2: 0.00 m")
    frame_label=Observable("Frame: 1 / 1")
    Label(panel[nextrow(),1], time_label,  fontsize=12, color=TEXT_COLOR)
    Label(panel[nextrow(),1], rmse_label,  fontsize=12, color=EST_COLOR)
    Label(panel[nextrow(),1], rmse_label2, fontsize=12, color=EST2_COLOR)
    Label(panel[nextrow(),1], frame_label, fontsize=12, color=TEXT_COLOR)

    function update_adaptive_limits!(idx)
        r=run_data[]
        i0=clamp(idx,2,r.n)
        p0=@view r.pos[1:i0,:]
        p1=@view est_state[].pos[1:min(i0,size(est_state[].pos,1)),:]
        p2=@view est_state2[].pos[1:min(i0,size(est_state2[].pos,1)),:]
        allp=vcat(p0,p1,p2)
        mn=vec(minimum(allp,dims=1)); mx=vec(maximum(allp,dims=1))
        c=(mn.+mx)./2; halfspan=(mx.-mn)./2
        halfspan=halfspan.*1.25 .+ 1e-2          
        s=maximum(halfspan)                       
        xlims!(ax3d, c[1]-s, c[1]+s); ylims!(ax3d, c[2]-s, c[2]+s); zlims!(ax3d, c[3]-s, c[3]+s)
    end

    full_t=Ref(Float32[0f0]); full_gt=Ref(Float32[0f0])
    full_e1=Ref(Float32[0f0]); full_e2=Ref(Float32[0f0]); has_gt=Ref(true)
    stat_x=Observable(Float32[0f0]); stat_gt=Observable(Float32[0f0])
    stat_e1=Observable(Float32[0f0]); stat_e2=Observable(Float32[0f0])
    l_gt =lines!(stat_ax, stat_x, stat_gt, color=GT_COLOR, linewidth=2, linestyle=:dash)
    l_e1 =lines!(stat_ax, stat_x, stat_e1, color=EST_COLOR, linewidth=2.5)
    l_e2 =lines!(stat_ax, stat_x, stat_e2, color=EST2_COLOR, linewidth=2.5)

    cur_key() =stat_keys[findfirst(==(stat_menu.selection[]), stat_options)]
    cur_comp()=(c=comp_menu.selection[]; c=="X" ? 1 : c=="Y" ? 2 : c=="Z" ? 3 : 0)

    function series_for(r, st, ba, bg, key, comp)
        an=("X","Y","Z")
        if key==:bias_acc || key==:bias_gyro
            real=key==:bias_acc ? r.bias_acc_det : r.bias_gyro_det
            pred=key==:bias_acc ? ba : bg
            gv = comp==0 ? Float32.(vec(sqrt.(sum(real.^2,dims=2)))) : Float32.(real[:,comp])
            ev = comp==0 ? Float32.(vec(sqrt.(sum(pred.^2,dims=2)))) : Float32.(pred[:,comp])
            return gv, ev, true
        elseif key==:err_pos || key==:err_vel
            realv=key==:err_pos ? r.pos : r.vel; estS=key==:err_pos ? st.pos : st.vel
            n=min(size(realv,1),size(estS,1))
            if comp==0
                ev=Float32.([norm(estS[i,:].-realv[i,:]) for i in 1:n]); return zeros(Float32,n), ev, false
            else
                return Float32.(realv[1:n,comp]), Float32.(estS[1:n,comp]), true
            end
        else # err_ori
            Egt=euler_matrix(r.quat); Eest=euler_matrix(st.quat); n=min(size(Egt,1),size(Eest,1))
            if comp==0
                ev=Float32.([norm(Eest[i,:].-Egt[i,:]) for i in 1:n]); return zeros(Float32,n), ev, false
            else
                return Float32.(Egt[1:n,comp]), Float32.(Eest[1:n,comp]), true
            end
        end
    end

    function recompute_stat_series()
        r=run_data[]; key=cur_key(); comp=cur_comp(); t=Float32.(r.t)
        an=("X","Y","Z"); cn=comp==0 ? "Total" : an[comp]
        gv1,ev1,hg = series_for(r, est_state[],  est_ba[],  est_bg[],  key, comp)
        gv2,ev2,_  = series_for(r, est_state2[], est_ba2[], est_bg2[], key, comp)
        n=min(length(ev1),length(ev2),length(t))
        full_t[]=t[1:n]; full_e1[]=ev1[1:n]; full_e2[]=ev2[1:n]
        full_gt[]=hg ? gv1[1:n] : zeros(Float32,n); has_gt[]=hg
        # etichete
        unit = key==:bias_acc ? "m/s²" : key==:bias_gyro ? "rad/s" : key==:err_vel ? "m/s" : key==:err_pos ? "m" : "rad"
        titlemap = Dict(:bias_acc=>"Bias acc",:bias_gyro=>"Bias gyro",:err_pos=>"Eroare poziție",
                        :err_vel=>"Eroare viteză",:err_ori=>"Eroare orientare")
        stat_ax.ylabel = "$(titlemap[key]) $cn ($unit)"
        stat_ax.title  = hg ? "$(titlemap[key]) $cn — GT vs metode" : "$(titlemap[key]) $cn — metoda 1 vs 2"
        l_gt.visible[]=has_gt[]
    end

    function sync_stat_to_frame(idx)
        N=length(full_t[]); m=clamp(idx,1,N)
        stat_x[]=Float32.(full_t[][1:m]); stat_e1[]=Float32.(full_e1[][1:m]); stat_e2[]=Float32.(full_e2[][1:m])
        stat_gt[]= has_gt[] ? Float32.(full_gt[][1:m]) : fill(NaN32, m)   # ← aceeași lungime, NaN nu se desenează
        if N>1
            xlims!(stat_ax, full_t[][1], full_t[][N])
            ydata=has_gt[] ? vcat(full_gt[],full_e1[],full_e2[]) : vcat(full_e1[],full_e2[])
            ylo,yhi=minimum(ydata),maximum(ydata); pad=0.05*(yhi-ylo)+1e-6
            ylims!(stat_ax, ylo-pad, yhi+pad)
        end
    end

    refresh_stat()=(recompute_stat_series(); sync_stat_to_frame(frame_idx[]))

    on(frame_idx) do idx
        r=run_data[]; st=est_state[]; st2=est_state2[]; i=min(idx,r.n)
        time_label[]=@sprintf("Timp: %.2f s", r.t[i])
        if i>1
            if i<=size(st.pos,1)
                rmse1=sqrt(mean(sum((st.pos[1:i,:].-r.pos[1:i,:]).^2,dims=2)))
                rmse_label[]=@sprintf("RMSE metoda 1: %.2f m", rmse1)
            end
            if i<=size(st2.pos,1)
                rmse2=sqrt(mean(sum((st2.pos[1:i,:].-r.pos[1:i,:]).^2,dims=2)))
                rmse_label2[]=@sprintf("RMSE metoda 2: %.2f m", rmse2)
            end
        end
        frame_label[]=@sprintf("Frame: %d / %d", i, r.n)
        update_adaptive_limits!(i)     # ← SCALĂ ADAPTIVĂ: axele urmăresc frontul curent
        sync_stat_to_frame(i)
    end

    function regenerate()
        ssym=shape_symbols[findfirst(==(shape_menu.selection[]), shape_options)]
        m1=method_symbols[findfirst(==(method_menu.selection[]), method_options)]
        m2=method_symbols[findfirst(==(method_menu2.selection[]), method_options)]
        println("Generez: $ssym | metoda1: $m1 | metoda2: $m2")
        new_run=generate_run(ssym)
        st1,a1,g1=compute_state(new_run,m1); st2,a2,g2=compute_state(new_run,m2)
        run_data[]=new_run
        est_state[]=st1; est_ba[]=a1; est_bg[]=g1
        est_state2[]=st2; est_ba2[]=a2; est_bg2[]=g2
        frame_idx[]=1
        update_adaptive_limits!(1)
        refresh_stat()
    end

    on(shape_menu.selection)   do _; regenerate(); end
    on(method_menu.selection)  do _; regenerate(); end
    on(method_menu2.selection) do _; regenerate(); end
    on(stat_menu.selection)    do _; refresh_stat(); end
    on(comp_menu.selection)    do _; refresh_stat(); end

    on(btn_start.clicks) do _
        is_playing[]=true
        @async begin
            while is_playing[]
                step=Int(speed_slider.value[]); n=run_data[].n
                nf=min(frame_idx[]+step,n); frame_idx[]=nf
                if nf>=n; is_playing[]=false; break; end
                sleep(0.03)
            end
        end
    end
    on(btn_pause.clicks) do _; is_playing[]=false; end
    on(btn_reset.clicks) do _; is_playing[]=false; frame_idx[]=1; end

    update_adaptive_limits!(1); refresh_stat(); display(fig)  
    println("\nSimulator pornit — comparație 2 metode. Alege traseu + 2 metode, apoi START.")
    println("Metoda 1 = portocaliu, Metoda 2 = verde. Axele 3D sunt extinse pt a vedea deriva.")
    return fig
end

launch_simulator()
println("Apasă Enter în terminal pentru a închide programul...")
readline()