using Base.Threads
using Parameters, FastGaussQuadrature, ForwardDiff , LinearAlgebra, DistributionsAD, Optim, CSV, DataFrames;
const SQRT_2 = √2;
const SQRT_PI = √π;
const PI = π;
const EPSILON = 1e-8;

filtered_data = CSV.read("src/r4/estimation_sample_007.csv", DataFrame);
# load the last best guess of parameters from the previous run
initparams = CSV.read("parameters/params1_nosq_002.csv", DataFrame);
params_nosq0 = initparams[:,1][1:end-1]
@with_kw mutable struct FertilityModel
    #=Dynamic discrete choice model
    Contains all the parameters for the base DDC model.
    Description of parameters =#
    fP_1 = 1.0*1e-3 # Productivity types
    fP_2 = 2.0*1e-3
    fC_1 = 1.0*1e-3 # Children preference types
    fC_2 = 2.0*1e-3
    T_last::Int64 = 39
    Tf::Int64 = 23

    # Utility Parameters
    β = 0.95
    ν1 = 1.0*1e-3
    ν2 = 1.0*1e-3
    ν3 = 0.5*1e-3
    ν4 = 0.0*1e-3
    ν5 = 1.0*1e-3
    ν6 = 1.0*1e-3
    ν7 = 1.0*1e-3
    ν8 = 1.0*1e-3
    ν9 = 1.0*1e-3
    γ1 = 1.0*1e-3
    γ2 = 1.0*1e-3
    γ3 = 1.0*1e-3
    γ4 = 1.0*1e-3
    γ5 = 1.0*1e-3
    γ6 = 1.0*1e-3
    γ7 = 1.0*1e-3
    γ8 = 1.0*1e-3

    # Wage parameters
    α_1 = 0.049943177212081*1e-3
    α_2 = 0.00422780547300336*1e-3
    α_3 = 0.2*1e-3
    α_4 = 0.0*1e-5
    α_5 = 0.3*1e-3
    α_6 = 0.0*1e-5    

    # Spouse earning parameters
    α_h0 = 2.32124368605255*1e-3
    α_h1 = 0.0157764327896929*1e-3
    α_h2 = -0.450152633415391*1e-5
    α_h3 = 0.0454586919914581*1e-3
    
    # Divorce and Marriage parameters
    λ_0M = 1.e-3
    λ_a1M = 1.e-3
    λ_a2M = 1.e-3
    λ_a3M = 1.e-3
    λ_2M = 1.e-3
    λ_3M = 1.e-3
    λ_4M = 0.e-3
    
    λ_0D = 1.e-3
    λ_a1D = 1.e-3
    λ_a2D = 1.e-3
    λ_a3D = 1.e-3
    λ_2D = 1.e-4
    λ_3D = 1.0e-3
    
    # Distribution parameters 
    σ_ww = 1.0*1e-1
    σ_yy = 1.0*1e-1
    σ_wy = 0.
end

function num_globals(Nq = 10)
    n1D, w1D = gausshermite(Nq)
    pruning_minwt = 1.05*(log(w1D[1]) + log(w1D[Nq÷2]))
    n2D = [(n1D[i], n1D[j]) for i = 1:Nq, j = 1:Nq]
    w2D = [w1D[i] * w1D[j] for i = 1:Nq, j = 1:Nq]
    # with pruning
    n2D_p = [(n1D[i], n1D[j]) for i = 1:Nq, j = 1:Nq if log(w1D[i])+log(w1D[j]) > pruning_minwt]
    w2D_p = [w1D[i] * w1D[j] for i = 1:Nq, j = 1:Nq if log(w1D[i])+log(w1D[j]) > pruning_minwt]
    return([n1D,w1D,n2D_p,w2D_p])
end

function get_state(t::Int,m::FertilityModel)
    # Define possible values for each variable
    ageM = [t];
    schooling = [0, 1] # 1 if college
    spouse = [0,1] # 1 if spouse is present
    num_child = [0, 1, 2, 3]
    min_age = max(0, t-m.Tf-1)
    if min_age >17
        num_child = [0]
        ageK = [0]
    elseif min_age>0
        ageK = [0;min_age:17;]
    else
        ageK = [min_age:17;]
    end
    Xf = [0:(t-1);]
    Xp = [0:(t-1);]
    productivity_types = [1] # 1 if high productivity
    fertility_types = [1] # 1 if high preference for fertility
    function state_validity(s)
        retval = true
        if (s[5]+s[6] > (t-1))
            # experience condition: total experience must be less than or equal to the number of years since 22
            retval = false
        end
        if s[3]==0 && s[4]>0
            # age of youngest child must be 0 if the number of children is 0
            retval = false
        elseif s[3]>0 && s[4]<min_age
            # in post fecund period the age of the child, if present, must be greater than the number of years since fecundity horizon
            retval = false
        end
        return retval
    end
    # Generate all possible state tuples that satisfy the constraints
    # S = [s for s in Iterators.product(ageM,productivity_types, fertility_types, schooling_M, schooling_F, num_child, ageK, Xf, Xp) if (s[end] + s[end-1] <= (t-1)) & (s[end-2] < (s[end-3] * 18))]
    # S = [s for s in Iterators.product(ageM, productivity_types, fertility_types,schooling_M,schooling_F, num_child,ageK, Xf,Xp) if (s[end]+s[end-1] <= (t-1)) && (s[end-2] <= ((s[end-3]==0)*0)) && (s[end-2] >= ((s[end-3]>0)*min_age))]
    S = [s for s in Iterators.product(ageM, schooling, num_child, ageK, Xf, Xp, spouse, productivity_types, fertility_types) if (state_validity(s) == true) ]
    return S
end

function update_state(st,n,h,marr_change = false)
    # extract the elements of the st vector
    ageM, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    # update the ageM by adding 1
    ageM_new = ageM + 1
    # generate part time full time status p,f as (h==1).*1 and (h==2).*1
    p = (h == 1)*1
    f = (h == 2)*1
    # update accumulated experience of both types Xf = Xf+f and Xp = Xp +p
    Xf_new = Xf + f
    Xp_new = Xp + p
    # update N and ageK based on input n
    ## Test logic
        # if n = 1, ageK <17 then update N = N+1 and ageK= 0
        # if n = 0, ageK <17, N>0 then update ageK = ageK+1
        # if n = 0, ageK <17, N=0 then update ageK = ageK
        # if n=0, ageK = 17, then update N to 0 and ageK to 0
        # if n=1, ageK = 17, then update N = 1 and ageK = 0
    #
    if n == 1
        if ageK < 17
            N_new = N + 1
            ageK_new = 0
        else
            N_new = 1
            ageK_new = 0
        end
    else
        if ageK < 17 && N > 0
            N_new = N
            ageK_new = ageK + 1
        elseif ageK < 17 && N == 0
            N_new = 0
            ageK_new = ageK
        elseif ageK == 17
            N_new = 0
            ageK_new = 0
        end
    end
    sp_new = ifelse(marr_change,1-sp,sp);
    st_new = (ageM_new, S, N_new, ageK_new, Xf_new, Xp_new, sp_new, fP_t, fC_t)
    return st_new
end

function utility_spec1(c, h, st,  m::FertilityModel)
    ageM, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    # N, S, age_M, age_K,
    # Unpack parameters
    @unpack β, ν1, ν2, ν3, ν4, ν5, ν6, ν7, ν8, ν9, γ1, γ2, γ3, γ4, γ5, γ6, γ7, γ8,fC_1,fC_2 = m
    f = (h==1)*1
    p = (h==2)*1
    fC = (fC_t==1) ? fC_2 : fC_1 #(1-fC_t)*fC_1 + fC_t*fC_2
    # Calculate utility components
    utility = c + ν1*p + ν2*f + ν3*N*S + ν4*N.^2 + ν5*c.*p + ν6*c.*f + ν7*c.*N + ν8*p.*N + ν9*f.*N + 
              (γ1 + γ2*S + γ3*ageM + γ4*ageK).*p + (γ5 + γ6*S + γ7*ageM + γ8*ageK).*f
    return utility
end

f_act1(x) = exp(x) / (1.0 + exp(x))
f_act2(x) = 1.0 / (1.0 + exp(-x))

function valT_single(m::FertilityModel, st,uw)
    t, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    @unpack fP_1, fP_2, fC_1, fC_2, T_last, Tf, β, ν1, ν2, ν3, ν4, ν5, ν6, ν7, ν8, ν9, γ1, γ2, γ3, γ4, γ5, γ6, γ7, γ8, α_1, α_2, α_3, α_4, α_5, α_6, α_h0, α_h1, α_h2, α_h3, λ_0M, λ_a1M, λ_a2M, λ_a3M, λ_2M, λ_3M, λ_4M, λ_0D, λ_a1D, λ_a2D, λ_a3D, λ_2D, σ_ww, σ_yy, σ_wy = m
    ϵw = sqrt(σ_ww)*uw
    fP = (fP_t==1) ? fP_2 : fP_1  #(1-fP_t)*fP_1 + fP_t*fP_2
    # fC = (fC_t==1) ? fC_2 : fC_1
    w = exp(fP + α_1*S + α_2*t + α_3*Xf + α_4*Xf^2 + α_5*Xp + α_6*Xp^2 + ϵw)
    Val_vec = [utility_spec1(h*20*w,h,st,m) for h in [0,1,2]]
    mV =  maximum(Val_vec)
    return mV+log(sum(exp.(Val_vec.-mV)))
    # return Val_vec
end

function valT_married(m::FertilityModel, st,uw,uh)
    t, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    @unpack fP_1, fP_2, fC_1, fC_2, T_last, Tf, β, ν1, ν2, ν3, ν4, ν5, ν6, ν7, ν8, ν9, γ1, γ2, γ3, γ4, γ5, γ6, γ7, γ8, α_1, α_2, α_3, α_4, α_5, α_6, α_h0, α_h1, α_h2, α_h3, λ_0M, λ_a1M, λ_a2M, λ_a3M, λ_2M, λ_3M, λ_4M, λ_0D, λ_a1D, λ_a2D, λ_a3D, λ_2D, σ_ww, σ_yy, σ_wy = m
    Σ = [σ_ww σ_wy; σ_wy σ_yy]
    L = cholesky(Σ).L
    Lu = L*[uw,uh]
    fP = (fP_t==1) ? fP_2 : fP_1  #(1-fP_t)*fP_1 + fP_t*fP_2
    # fC = (fC_t==1) ? fC_2 : fC_1
    w = exp(fP + α_1*S + α_2*t + α_3*Xf + α_4*Xf^2 + α_5*Xp + α_6*Xp^2 + Lu[1])
    earnh = α_h0 + α_h1*t + α_h2*t^2 + α_h3*S + Lu[2]
    Val_vec = [utility_spec1(h*20*w+earnh,h,st,m) for h in [0,1,2]]
    mV =  maximum(Val_vec)
    return mV+log(sum(exp.(Val_vec.-mV)))
    # return Val_vec
end


function EmaxT(m::FertilityModel,st,gvars)
    t, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    n1D,w1D,Nxy,Wxy = gvars
    # Nxy,Wxy = gvars
    # requires integration of valT over the shocks
    # requires sqrt(2) for each coordinate and division by sqrt(π)^(number of variables)
    if sp==1
        # if married (sp = 1) we integrate over both the shocks for w and sp_earn
        return sum(valT_married(m,st,sqrt(2)*nxy[1],sqrt(2)*nxy[2])*Wxy[i] for (i,nxy) in enumerate(Nxy))/π
    else
        # if single (sp = 0) we integrate over the shock for w
        return sum(valT_single(m,st,sqrt(2)*nx)*w1D[i] for (i,nx) in enumerate(n1D))/sqrt(π)
    end
end

function valtf_single(m::FertilityModel, st, uw, V_n)
    t, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    @unpack fP_1, fP_2, fC_1, fC_2, T_last, Tf, β, ν1, ν2, ν3, ν4, ν5, ν6, ν7, ν8, ν9, γ1, γ2, γ3, γ4, γ5, γ6, γ7, γ8, α_1, α_2, α_3, α_4, α_5, α_6, α_h0, α_h1, α_h2, α_h3, λ_0M, λ_a1M, λ_a2M, λ_a3M, λ_2M, λ_3M, λ_4M, λ_0D, λ_a1D, λ_a2D, λ_a3D, λ_2D, σ_ww, σ_yy, σ_wy = m

    fP = fP_t == 1 ? fP_2 : fP_1
    # fC = fC_t == 1 ? fC_2 : fC_1

    fecund = (t <= Tf) && (N < 3) ? true : false
    N_opts = fecund ? [0, 1] : [0]

    marr_logit = λ_0M + λ_a1M * (t < 12 ? 1 : 0) + λ_a2M * (t >= 12 ? 1 : 0) + λ_2M * Xf + λ_3M * Xp
    marr_prob = marr_logit > 0 ? f_act2(marr_logit) : f_act1(marr_logit)

    # ϵw = sqrt(σ_ww)*uw
    w = exp(fP + α_1 * S + α_2 * t + α_3 * Xf + α_4 * Xf^2 + α_5 * Xp + α_6 * Xp^2 + sqrt(σ_ww)*uw)

    Val_vec = [utility_spec1(h * 20 * w, h, st, m) + marr_prob * V_n[update_state(st, n, h, true)] + (1.0 - marr_prob) * V_n[update_state(st, n, h, false)] for h in [0, 1, 2] for n in N_opts]

    mV = maximum(Val_vec)
    return mV + log(sum(exp.(Val_vec .- mV)))
end


function valtf_married(m::FertilityModel, st, uw, uh, V_n)
    t, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    @unpack fP_1, fP_2, fC_1, fC_2, T_last, Tf, β, ν1, ν2, ν3, ν4, ν5, ν6, ν7, ν8, ν9, γ1, γ2, γ3, γ4, γ5, γ6, γ7, γ8, α_1, α_2, α_3, α_4, α_5, α_6, α_h0, α_h1, α_h2, α_h3, λ_0M, λ_a1M, λ_a2M, λ_a3M, λ_2M, λ_3M, λ_4M, λ_0D, λ_a1D, λ_a2D, λ_a3D, λ_2D, σ_ww, σ_yy, σ_wy = m

    fecund = (t <= Tf) && (N < 3) ? true : false
    N_opts = fecund ? [0, 1] : [0]

    div_logit = λ_0D * S + (t < 12 ? λ_a1D : (t < 22 ? λ_a2D : λ_a3D))
    div_prob = div_logit > 0 ? f_act2(div_logit) : f_act1(div_logit)

    # Lu = [sqrt(σ_ww) * uw, sqrt(σ_yy) * uh]

    fP = fP_t == 1 ? fP_2 : fP_1
    w = exp(fP + α_1 * S + α_2 * t + α_3 * Xf + α_4 * Xf^2 + α_5 * Xp + α_6 * Xp^2 + sqrt(σ_ww)*uw)
    earnh = α_h0 + α_h1 * t + α_h2 * t^2 + α_h3 * S + sqrt(σ_yy) * uh

    Val_vec = [utility_spec1(h * 20 * w, h, st, m) + div_prob * V_n[update_state(st, n, h, true)] + (1.0 - div_prob) * V_n[update_state(st, n, h, false)] for h in [0, 1, 2] for n in N_opts]

    mV = maximum(Val_vec)
    return mV + log(sum(exp.(Val_vec .- mV)))
end

function Emaxtf(m::FertilityModel, st, gvars, EV_n)
    t, S, N, ageK, Xf, Xp, sp, fP_t, fC_t = st
    n1D, w1D, Nxy, Wxy = gvars

    if sp == 1
        # If married (sp = 1), integrate over both the shocks for w and sp_earn
        return sum(valtf_married(m,st,SQRT_2*nxy[1],SQRT_2*nxy[2], EV_n)*Wxy[i] for (i,nxy) in enumerate(Nxy))/PI
    else
        # If single (sp = 0), integrate over the shock for w
        return sum(valtf_single(m,st,SQRT_2*nx, EV_n)*w1D[i] for (i,nx) in enumerate(n1D))/SQRT_PI
    end
end

function fullrun(m::FertilityModel)
    Ts = collect(m.T_last:-1:1)
    Sts = [get_state(t, m) for t in Ts]
    EVs = Vector{Dict}(undef, length(Sts))
    for (i, S) in enumerate(Sts)
        if Ts[i] == m.T_last
            results = Array{Any}(undef, length(S))
            @threads for j in eachindex(S)
                s = S[j]
                results[j] = (s, EmaxT(m, s, gvars))
            end
            EVs[i] = Dict(results)
        else
            results = Array{Any}(undef, length(S))
            @threads for j in eachindex(S)
                s = S[j]
                results[j] = (s, Emaxtf(m, s, gvars, EVs[i - 1]))
            end
            EVs[i] = Dict(results)
        end
    end
    return EVs
end

function ind_marrLL(row,m::FertilityModel, verbose = false)
    LL = 0.0
    @unpack fP_1, fP_2, fC_1, fC_2, T_last, Tf, β, ν1, ν2, ν3, 
    ν4, ν5, ν6, ν7, ν8, ν9, γ1, γ2, γ3, γ4, γ5, γ6, γ7, 
    γ8, α_1, α_2, α_3, α_4, α_5, α_6, α_h0, α_h1, α_h2, 
    α_h3, λ_0M, λ_a1M, λ_a2M, λ_a3M, λ_2M, λ_3M, λ_4M, 
    λ_0D, λ_a1D, λ_a2D, λ_a3D, λ_2D, σ_ww, σ_yy, σ_wy = m;
    # n1D,w1D,Nxy,Wxy = gvars;
    # Σ = [σ_ww σ_wy; σ_wy σ_yy];
    # L = cholesky(Σ).L;
    # Lu = L*[uw,uh]
    fP = fP_2;
    fC = fC_2;
    S = row.coll;
    N = row.nchild;
    sp = row.husb;
    @unpack Xf,Xp,t = row;
    if(row.nchild==0)
        ageK = 0
    else
        ageK = tryparse(Int,row.agey)
    end
    marr_logit = λ_0M+ λ_a1M*ifelse(t<12,1,0)+ λ_a2M*ifelse(t>=12,1,0)+ λ_2M*Xf+ λ_3M*Xp
    marr_prob = ifelse(marr_logit>0,f_act2(marr_logit),f_act1(marr_logit))
    div_logit = λ_0D*S+ ifelse(t<12, λ_a1D,ifelse(t<22,λ_a2D,λ_a3D))
    div_prob = ifelse(div_logit>0,f_act2(div_logit), f_act1(div_logit))
    p_marrchange = ifelse(sp==1,div_prob,marr_prob);
    marrchange = ifelse(sp==1,row.outsplit,row.outmarr)
    lpmarrchange = marrchange*log(p_marrchange)+(1.0-marrchange)*log(1.0-p_marrchange)
    (verbose) && (isnan(lpmarrchange) || isinf(lpmarrchange)) && (println("lpmarrchange = ", lpmarrchange)," sp = ", sp, " p_marrchange = ", p_marrchange, " marrchange = ", marrchange)
    LL = LL+lpmarrchange;
    #print("\nLL = ", LL)
    return LL
end

function indrow_LL(row,m::FertilityModel, EVs, verbose = false)
    LL = 0.0
    @unpack fP_1, fP_2, fC_1, fC_2, T_last, Tf, β, ν1, ν2, ν3, 
    ν4, ν5, ν6, ν7, ν8, ν9, γ1, γ2, γ3, γ4, γ5, γ6, γ7, 
    γ8, α_1, α_2, α_3, α_4, α_5, α_6, α_h0, α_h1, α_h2, 
    α_h3, λ_0M, λ_a1M, λ_a2M, λ_a3M, λ_2M, λ_3M, λ_4M, 
    λ_0D, λ_a1D, λ_a2D, λ_a3D, λ_2D, σ_ww, σ_yy, σ_wy = m;
    n1D,w1D,Nxy,Wxy = gvars;
    Σ = [σ_ww σ_wy; σ_wy σ_yy];
    L = cholesky(Σ).L;
    # Lu = L*[uw,uh]
    fP = fP_2;
    fC = fC_2;
    S = row.coll;
    N = row.nchild;
    sp = row.husb;
    @unpack Xf,Xp,t = row;
    if(row.nchild==0)
        ageK = 0
    else
        ageK = tryparse(Int,row.agey)
    end
    Elogw = fP + α_1*S + α_2*t + α_3*Xf + α_4*Xf^2 + α_5*Xp + α_6*Xp^2;
    Eearn = α_h0 + α_h1*t + α_h2*t^2 + α_h3*S
    # whether wage observed
    # whether married
    fecund = ifelse(t<=Tf && N<3,1,0);
    Nopts = ifelse(fecund==1,[0,1],[0]);
    st = (t,S,N, ageK,Xf,Xp, sp, 1,1);
    f_act1(x) = exp(x)/(1.0 + exp(x));
    f_act2(x) = 1.0/(1.0 + exp(-x));
    marr_logit = λ_0M+ λ_a1M*ifelse(t<12,1,0)+ λ_a2M*ifelse(t>=12,1,0)+ λ_2M*Xf+ λ_3M*Xp
    marr_prob = ifelse(marr_logit>0,f_act2(marr_logit),f_act1(marr_logit))
    div_logit = λ_0D+ λ_a1D*ifelse(t<12,1,0)+ λ_a2D*ifelse(t>=12,1,0)
    div_prob = ifelse(div_logit>0,f_act2(div_logit), f_act1(div_logit))
    p_marrchange = ifelse(sp==1,div_prob,marr_prob);
    marrchange = ifelse(sp==1,row.outsplit,row.outmarr)
    lpmarrchange = marrchange*log(p_marrchange)+(1.0-marrchange)*log(1.0-p_marrchange)
    (verbose) && (isnan(lpmarrchange) || isinf(lpmarrchange)) && (println("lpmarrchange = ", lpmarrchange)," sp = ", sp, " p_marrchange = ", p_marrchange, " marrchange = ", marrchange)
    # LL = LL+lpmarrchange;
    # whether wage observed
    w = tryparse(Float64, row.wage);
    # sp-earn
    if sp==0
        earnh = 0.0
    else
        earnh = tryparse(Float64, row.earn)
        # ϵy = earn - Elogearn
    end
    # Here we add only the known part of the logpdfs to LL
    Etf = EVs[T_last-row.t];
    if(!isnothing(w))
        #  No spouse, wage not observed
        ϵw = log(w) - Elogw
        if sp==0
            LL = LL + DistributionsAD.logpdf(DistributionsAD.Normal(0.0,σ_ww),ϵw)
            Val_vec = [[utility_spec1(h*20*w+earnh,h,st,m)+ p_marrchange* Etf[update_state(st,n,h, true)] + (1.0-p_marrchange)* Etf[update_state(st,n,h, false)] for h in [0,1,2]] for n in Nopts];
            mV =  maximum(maximum(Val_vec));
            logits = [exp.(Vrow.-mV) for Vrow in Val_vec]
            logits_stabilized = [lrow .+ EPSILON for lrow in logits]
            plogits = [lrow ./ sum(sum(logits_stabilized)) for lrow in logits_stabilized]
            # print("\nLogit = ",logits[row.outn+1][row.outh+1])
            lpchoice = log.(plogits[row.outn+1][row.outh+1])
            (verbose) && (isnan(lpchoice) || isinf(lpchoice)) && (print("ϵw = ", ϵw, "\t lpchoice = ", lpchoice, "\n"))
            LL = LL + lpchoice
        else
            if !isnothing(earnh)
                ϵy = earnh/52.0 - Eearn
                Lpdf_known = DistributionsAD.logpdf(DistributionsAD.MvNormal([0.0,0.0],Σ),[ϵw, ϵy]);
                (verbose) && (isnan(Lpdf_known) || isinf(Lpdf_known)) && (print("ϵw = ", ϵw, "\t ϵy = ", ϵy, "\t Σ = ", Σ, "\t Lpdf_known = ", Lpdf_known, "\n"))
                LL = LL + DistributionsAD.logpdf(DistributionsAD.MvNormal([0.0,0.0],Σ),[ϵw, ϵy])
                # print("\nLL = ",LL)
                Val_vec = [[utility_spec1(h*20*w+earnh,h,st,m)+ p_marrchange* Etf[update_state(st,n,h, true)] + (1.0-p_marrchange)* Etf[update_state(st,n,h, false)] for h in [0,1,2]] for n in Nopts];
                mV =  maximum(maximum(Val_vec));
                logits = [exp.(Vrow.-mV) for Vrow in Val_vec];
                logits_stabilized = [lrow .+ EPSILON for lrow in logits]
                plogits = [lrow ./ sum(sum(logits_stabilized)) for lrow in logits_stabilized]
                # print("\nLogit = ",logits[row.outn+1][row.outh+1])
                lpchoice = log.(plogits[row.outn+1][row.outh+1])
                (verbose) && (isnan(lpchoice) || isinf(lpchoice)) && (print("logits = ", logits, "\t lpchoice = ", lpchoice, "Val_vec = ", Val_vec, "\n"))
                LL = LL + lpchoice
            else
                # ! FIRST INTEGRAL CASE: Spouse present, wage observed, earn not observed
                # FIXME: Test this case
                # Write here the conditional distribution using the Schur complement
                σycond = σ_yy - σ_wy*(σ_ww)^-1*σ_wy;
                μycond = σ_wy*(σ_ww)^-1*ϵw;
                # cond_dist = DistributionsAD.Normal(μycond,σycond);
                Lpdf_known = DistributionsAD.logpdf(DistributionsAD.Normal(0.0,sqrt(σ_ww)),ϵw)
                (verbose) && (isnan(Lpdf_known) || isinf(Lpdf_known)) && (print("ϵw = ", ϵw, "\t Lpdf_known = ", Lpdf_known, "\n"))
                LL += Lpdf_known
                ## integral over quadrature here
                # integrand
                function integrand1(m, st, uh)
                    earnh = Eearn+μycond+sqrt(σycond)*uh
                    Val_vec = [[utility_spec1(h*20*w+earnh,h,st,m)+ p_marrchange* Etf[update_state(st,n,h, true)] + (1.0-p_marrchange)* Etf[update_state(st,n,h, false)] for h in [0,1,2]] for n in Nopts];
                    mV =  maximum(maximum(Val_vec));
                    logits = [exp.(Vrow.-mV) for Vrow in Val_vec]
                    logits_stabilized = [lrow .+ EPSILON for lrow in logits]
                    plogits = [lrow ./ sum(sum(logits_stabilized)) for lrow in logits_stabilized]
                    return plogits[row.outn+1][row.outh+1]
                end
                lprob = log(sum(integrand1(m,st,sqrt(2)*nx)*w1D[i] for (i,nx) in enumerate(n1D))/sqrt(π));
                (verbose) && (isnan(lprob) || isinf(lprob)) && (print("lprob = ", lprob, "\n"))
                LL += lprob;
                if verbose
                    # print("\nNOTESpouse, wage observed, earn not observed")
                    # print("\tHours choice =", row.outh)
                    # print("\tLL = ",LL)
                end
            end
        end
    else
        ## * wage not observed
        if sp==0
            # ! Second Integral case: No spouse, wage not observed
            ## integral over quadrature here
            # integrand
            function integrand2(m, st, uh)
                w = exp(Elogw+sqrt(σ_ww)*uh)
                Val_vec = [[utility_spec1(h*20*w+earnh,h,st,m)+ p_marrchange* Etf[update_state(st,n,h, true)] + (1.0-p_marrchange)* Etf[update_state(st,n,h, false)] for h in [0,1,2]] for n in Nopts];
                mV =  maximum(maximum(Val_vec));
                logits = [exp.(Vrow.-mV) for Vrow in Val_vec]
                logits_stabilized = [lrow .+ EPSILON for lrow in logits]
                plogits = [lrow ./ sum(sum(logits_stabilized)) for lrow in logits_stabilized]
                return plogits[row.outn+1][row.outh+1]
            end
            lprob = log(sum(integrand2(m,st,sqrt(2)*nx)*w1D[i] for (i,nx) in enumerate(n1D))/sqrt(π));
            (verbose) && (isnan(lprob) || isinf(lprob)) && (print("lprob = ", lprob, "\n"))
            LL += lprob;
        else
            if !isnothing(earnh)
                # ! Third Integral case: Spouse present, wage not observed, earn observed
                ϵy = (earnh/52.0 - Eearn);
                Lpdf_known = DistributionsAD.logpdf(DistributionsAD.Normal(),ϵy/sqrt(σ_yy))
                (verbose) && (isnan(Lpdf_known) || isinf(Lpdf_known)) && (print("ϵy = ", ϵy, "\t Lpdf_known = ", Lpdf_known, "\n"))
                LL = LL + Lpdf_known
                σwcond = σ_ww - σ_wy*(σ_yy)^-1*σ_wy;
                μwcond = σ_wy*(σ_yy)^-1*ϵy;
                # cond_dist = DistributionsAD.Normal(μwcond,σwcond);
                ## integral over quadrature here
                # integrand
                function integrand3(m, st, uh)
                    w = exp(Elogw+μwcond+sqrt(σwcond)*uh)
                    Val_vec = [[utility_spec1(h*20*w+earnh,h,st,m)+ p_marrchange* Etf[update_state(st,n,h, true)] + (1.0-p_marrchange)* Etf[update_state(st,n,h, false)] for h in [0,1,2]] for n in Nopts];
                    mV =  maximum(maximum(Val_vec));
                    logits = [exp.(Vrow.-mV) for Vrow in Val_vec]
                    logits_stabilized = [lrow .+ EPSILON for lrow in logits]
                    plogits = [lrow ./ sum(sum(logits_stabilized)) for lrow in logits_stabilized]
                    return plogits[row.outn+1][row.outh+1]
                end
                lprob = log(sum(integrand3(m,st,sqrt(2)*nx)*w1D[i] for (i,nx) in enumerate(n1D))/sqrt(π));
                (verbose) && (isnan(lprob) || isinf(lprob)) && (print("lprob = ", lprob, "\n"))
                LL += lprob;
            else
                #  Fourth (Double) Integral case: Spouse present, wage not observed, earn not observed
                function integrand4(m,st,uw,uh)
                    Lu = L*[uw,uh]
                    w = exp(Elogw+Lu[1])
                    y = Eearn+Lu[2]
                    Val_vec = [[utility_spec1(h*20*w+y,h,st,m)+ p_marrchange* Etf[update_state(st,n,h, true)] + (1.0-p_marrchange)* Etf[update_state(st,n,h, false)] for h in [0,1,2]] for n in Nopts];
                    mV =  maximum(maximum(Val_vec));
                    logits = [exp.(Vrow.-mV) for Vrow in Val_vec]
                    logits_stabilized = [lrow .+ EPSILON for lrow in logits]
                    plogits = [lrow ./ sum(sum(logits_stabilized)) for lrow in logits_stabilized]
                    return plogits[row.outn+1][row.outh+1]
                end
                # lprob = log(sum(integrand(m,st,sqrt(2)*nx,sqrt(2)*ny)*w1D[i]*w1D[j] for (i,nx) in enumerate(n1D), (j,ny) in enumerate(n1D))/π); # for integral without pruning
                lprob = log(sum(integrand4(m,st,sqrt(2)*nxy[1],sqrt(2)*nxy[2])*Wxy[i] for (i,nxy) in enumerate(Nxy))/π); # for integral with pruning
                (verbose) && (isnan(lprob) || isinf(lprob)) && (print("lprob = ", lprob, "\n"))
                LL += lprob;
            end
        end
    end
    # print("\nLL = ", LL)
    return LL
end

function fertmodelLL_nosq_marr(parameters)
    fm = FertilityModel(
        λ_a1M = parameters[1],
        λ_a2M = parameters[2],
        λ_a3M = 0.0,
        λ_2M = parameters[3],
        λ_3M = parameters[4],
        λ_4M = 0.0,
        λ_0D = parameters[5],
        λ_a1D = parameters[6],
        λ_a2D = parameters[7],
        λ_a3D = parameters[8],
        λ_2D = 0.0,
    )
    # EVs = fullrun(fm);
    fullLL = 0.0
    for row in eachrow( filtered_data )
        ir_LL = ind_marrLL(row,fm, true)
        fullLL += ir_LL
    end
    return -1.0*fullLL
    # return -1.0*fullLL/size(filtered_data,1)
    #-1 so that we minimize the negative log likelihood
end

# ! Below needs a global names res_marr of solution class with field minimizer
function fertmodelLL_nosq(parameters)
    sig_ww = exp(parameters[23])
    sig_yy = exp(parameters[24])
    # rho_wy = (exp(2.0*parameters[41])-1.0)/(exp(2.0*parameters[41])+1.0)
    # sig_wy = rho_wy*sqrt(sig_ww*sig_yy)
    fm = FertilityModel(
        fP_2 = parameters[1],
        fC_2 = 0.0,
        ν1 = 0.0,
        ν2 = 0.0,
        ν3 = parameters[2],
        ν4 = 0.0*1e-3,  # Default value from the struct definition
        ν5 = parameters[3],
        ν6 = parameters[4],
        ν7 = parameters[5],
        ν8 = parameters[6],
        ν9 = parameters[7],
        γ1 = parameters[8],
        γ2 = parameters[9],
        γ3 = parameters[10],
        γ4 = parameters[11],
        γ5 = parameters[12],
        γ6 = parameters[13],
        γ7 = parameters[14],
        γ8 = parameters[15],
        α_1 = parameters[16],
        α_2 = parameters[17],
        α_3 = parameters[18],
        α_4 = 0.0*1e-5,  # Default value from the struct definition
        α_5 = parameters[19],
        α_6 = 0.0*1e-5,  # Default value from the struct definition
        α_h0 = parameters[20],
        α_h1 = parameters[21],
        α_h2 = 0.0,
        α_h3 = parameters[22],
        λ_a1M = res_marr.minimizer[1],
        λ_a2M = res_marr.minimizer[2],
        λ_a3M = 0.0,
        λ_2M = res_marr.minimizer[3],
        λ_3M = res_marr.minimizer[4],
        λ_4M = 0.0,
        λ_0D = res_marr.minimizer[5],
        λ_a1D = res_marr.minimizer[6],
        λ_a2D = res_marr.minimizer[7],
        λ_a3D = res_marr.minimizer[8],
        λ_2D = 0.0,
        σ_ww = sig_ww,
        σ_yy = sig_yy,
        σ_wy = 0.0
    )
    EVs = fullrun(fm);
    fullLL = 0.0
    for row in eachrow( filtered_data )
        ir_LL = indrow_LL(row,fm, EVs, true)
        fullLL += ir_LL
    end
    print("Total Loss = ",-1*fullLL,"\n")
    return -1.0*fullLL
    #-1 so that we minimize the negative log likelihood
end


















## Testing - simple tests
# fm1 = FertilityModel();
gvars = num_globals(10);
# S39 = get_state(fm1.T_last,fm1);
# # Test the value functions for married, single
# valT_married(fm1,S39[1],0.0,0.0)
# valT_single(fm1,S39[1],0.0)
# # Emax function for single state vector
# print("\nTime for one EmaxT iteration:")
# @time EmaxT(fm1,S39[1],gvars)
# print("\nTime for one EmaxT iteration, 2nd run:")
# @time EmaxT(fm1,S39[2],gvars)

# Testing the full last period solving time
# Non parallel computation time for comparison
# print("\nTime for all EmaxT iterations (serial):")
# @time EV39 = Dict(s => EmaxT(fm1,s,gvars) for s in S39)
### Test to show parallelization is working
# function EV_S(S,m)
#     results = Array{Any}(undef, length(S))
#     @threads for j in eachindex(S)
#         s = S[j]
#         results[j] = (s, EmaxT(m, s, gvars))
#     end
#     return Dict(results)
# end
# print("\nTime for all EmaxT iterations (parallel):\t")
# @time EV39 = EV_S(S39,fm1);


# print("Full solution time:\t")
# @time EVs = fullrun(fm1);

# print("\n")


params1_nosq_marr = [-2.451530904485376, -2.861783624451035, 0.0026974881900206395, -0.02465260499082496, -0.9581658339932975, -2.8801862247979577, -2.4996060588734044, -2.648396682059441];
res_marr = Optim.optimize(fertmodelLL_nosq_marr, params1_nosq_marr, NelderMead(),Optim.Options(f_calls_limit = 100))
res_marr = optimize(fertmodelLL_nosq_marr, res_marr.minimizer, Newton(), autodiff=:forward)

print("Minimizer for marr, divorced, widowed = ", res_marr.minimizer, "\n")
# save the results as a csv file
res_marrdf = DataFrame(marr_div_coeffs = res_marr.minimizer)
CSV.write("parameters/res_marr.csv", res_marrdf)

# gradmarr = ForwardDiff.gradient(fertmodelLL_nosq_marr,res_marr.minimizer)

hessmarr = ForwardDiff.hessian(fertmodelLL_nosq_marr,res_marr.minimizer);
# print("Hessian for marr, divorced, widowed = ", hessmarr, "\n")

se_vec_marr = sqrt.(diag(inv(hessmarr))./size(filtered_data,1))
res_marrdfSE = DataFrame(SEvec = se_vec_marr)
CSV.write("parameters/res_marrSE.csv", res_marrdfSE)


print("Standard errors for marr, divorced, widowed = ", se_vec_marr, "\n")

print("\n")


print("Time taken for LL computation:")
@time LL1 = fertmodelLL_nosq(params_nosq0)

print("\nLoss Likelihood (initial) = ", LL1, "\n")

print("\n")

# Now we will do Nelder Mead Optimization of the fertmodelLL_nosq function
# We first define the options for the optimization
options = Optim.Options(f_tol = 1e-2, x_tol = 1e-3, f_calls_limit = 150, show_trace = true)
# Our initial guess of parameters was loaded from data

res_nosq = Optim.optimize(fertmodelLL_nosq, params_nosq0, NelderMead(),options)

print("Minimizer for marr, divorced, widowed = ", res_nosq.minimizer, "\n")

# Exporting the results as dataframes
df_res_nosq = DataFrame(p_vec = res_nosq.minimizer)

# Write the dataframes to csv file
CSV.write("parameters/res_nosq_001.csv", df_res_nosq)