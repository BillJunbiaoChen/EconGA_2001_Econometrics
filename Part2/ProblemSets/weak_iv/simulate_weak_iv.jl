using LinearAlgebra, Random, Distributions, Plots, StatsBase

# ------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------
Random.seed!(123)
K_vec = [2, 10, 100, 200]
n_vec = [20, 200, 2000]
R = 20

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
function ols_iv_and_se(y, x, Z)
    n = length(y)
    X = hcat(ones(n), x)              # n×2
    Zfull = hcat(ones(n), Z)          # n×(K+1)

    # OLS: beta = (X'X)^{-1} X'y
    β_ols = (X'X) \ (X'y)

    # 2SLS
    Xhat = Zfull * ((Zfull'Zfull) \ (Zfull'X))
    β_iv = (Xhat'Xhat) \ (Xhat'y)

    # Residuals using structural regressors X
    uhat = y - X * β_iv

    # Usual homoskedastic asymptotic variance:
    # Var(β̂) = σ̂^2 * (X' P_Z X)^{-1}, σ̂^2 = (û'û)/(n - p)
    p = size(X, 2)                    # p=2
    σ2hat = (uhat'uhat) / (n - p)
    XPZX = X'Xhat                     # X'P_ZX

    V = σ2hat * inv(XPZX)
    se_slope = sqrt(V[2, 2])

    return β_ols[2], β_iv[2], se_slope
end

function simulate_one_dgp(n, K)
    z1 = randn(n)
    z2 = randn(n)
    e1 = randn(n)
    e2 = randn(n)
    e3 = randn(n)

    x = z1 .+ z2 .+ e1 .+ e2
    y = 1 .+ x .+ e1 .+ e3

    if K == 2
        Z = hcat(z1, z2)
    else
        z_junk = randn(n, K - 2)
        Z = hcat(z1, z2, z_junk)
    end

    return y, x, Z
end

# ------------------------------------------------------------
# Run simulations: store betas and SEs for each (n,K)
# ------------------------------------------------------------
results = Dict{Tuple{Int,Int}, Dict{Symbol, Vector{Float64}}}()

for n in n_vec, K in K_vec
    βols = ones(R)
    βiv  = ones(R)
    se   = ones(R)

    for r in 1:R
        println("Working on n=$n, K=$K, replication $r ...")
        y, x, Z = simulate_one_dgp(n, K)
        βols[r], βiv[r], se[r] = ols_iv_and_se(y, x, Z)
    end

    results[(n, K)] = Dict(:ols => βols, :iv => βiv, :se_iv => se)
end

# ------------------------------------------------------------
# Build label order once (same order used in both figures)
# ------------------------------------------------------------
labels = String[]
for n in n_vec, K in K_vec
    push!(labels, "n=$(n), K=$(K)")
end
xpos = collect(1:length(labels))

# ------------------------------------------------------------
# Plot style (bigger fonts)
# ------------------------------------------------------------
FS_TICK  = 12
FS_GUIDE = 16
FS_LEG   = 13
FS_TITLE = 18

default(;
    guidefontsize = FS_GUIDE,
    tickfontsize  = FS_TICK,
    legendfontsize= FS_LEG,
    titlefontsize = FS_TITLE,
    linewidth     = 2,
)

mkpath("images")

# ------------------------------------------------------------
# FIGURE 1: OLS vs IV (medians) with IQR error bars (vertical bars)
# ------------------------------------------------------------
med_ols = Float64[]
iqr_ols = Float64[]
med_iv  = Float64[]
iqr_iv  = Float64[]

for n in n_vec, K in K_vec
    βols = results[(n, K)][:ols]
    βiv  = results[(n, K)][:iv]

    qols = quantile(βols, [0.25, 0.5, 0.75])
    qiv  = quantile(βiv,  [0.25, 0.5, 0.75])

    push!(med_ols, qols[2]); push!(iqr_ols, qols[3] - qols[1])
    push!(med_iv,  qiv[2]);  push!(iqr_iv,  qiv[3] - qiv[1])
end

offset = 0.18
xpos_ols = xpos .- offset
xpos_iv  = xpos .+ offset

half_iqr_ols = iqr_ols ./ 2
half_iqr_iv  = iqr_iv  ./ 2

p_beta = plot(
    size=(1500, 750),
    xlabel="(n, K)",
    ylabel="Estimated slope (β̂ on x)",
    xticks=(xpos, labels),
    xrotation=35,
    legend=:topright,
    framestyle=:box,
    bottom_margin=14Plots.mm,
    left_margin=12Plots.mm,
)

bar!(p_beta, xpos_ols, med_ols;
    label="OLS median",
    alpha=0.65,
    bar_width=0.32,
    linecolor=:transparent,
    color=:steelblue
)
bar!(p_beta, xpos_iv, med_iv;
    label="2SLS median",
    alpha=0.65,
    bar_width=0.32,
    linecolor=:transparent,
    color=:darkorange
)

# IQR whiskers
scatter!(p_beta, xpos_ols, med_ols; yerror=half_iqr_ols, marker=:none, linecolor=:steelblue, label="")
scatter!(p_beta, xpos_iv,  med_iv;  yerror=half_iqr_iv,  marker=:none, linecolor=:darkorange, label="")

hline!(p_beta, [1.0], linestyle=:dash, linecolor=:black, label="true β=1")

display(p_beta)
savefig(p_beta, "images/comparison_ols_iv.png")

# ------------------------------------------------------------
# FIGURE 2: Mean asymptotic 2SLS SE for each (n,K)
# ------------------------------------------------------------
mean_se_iv = Float64[]
for n in n_vec, K in K_vec
    push!(mean_se_iv, mean(results[(n, K)][:se_iv]))
end

p_se = bar(
    xpos, mean_se_iv;
    label="Mean asymptotic 2SLS SE",
    color=:mediumpurple4,
    alpha=0.75,
    linecolor=:transparent,
    bar_width=0.7,
    size=(1500, 750),
    xlabel="(n, K)",
    ylabel="Mean asymptotic SE of 2SLS slope",
    xticks=(xpos, labels),
    xrotation=35,
    framestyle=:box,
    legend=:topright,
    bottom_margin=14Plots.mm,
    left_margin=12Plots.mm,
)

display(p_se)
savefig(p_se, "images/mean_2sls_se_by_nK.png")
