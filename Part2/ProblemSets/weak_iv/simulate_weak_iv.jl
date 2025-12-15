using LinearAlgebra, Random, Distributions, Plots, StatsBase

# ------------------------------------------------------------
# Simulation settings
# ------------------------------------------------------------
Random.seed!(123)                      # reproducible
K_vec = [2, 10, 100, 200]
n_vec = [20, 200, 2000]
R = 20                                # replications

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

function ols_2sls_slopes(y, x, Z)
    n = length(y)
    X = hcat(ones(n), x)               # include intercept
    Zfull = hcat(ones(n), Z)           # include intercept in first stage

    # OLS: beta = (X'X)^{-1} X'y
    β_ols = (X'X) \ (X'y)

    # IV
    Xhat = Zfull * ((Zfull'Zfull) \ (Zfull'X))
    β_iv  = (Xhat'Xhat) \ (Xhat'y)     # since second stage is y on Xhat

    return β_ols[2], β_iv[2]           # return slope on x
end


"""
Simulate one dataset given n, K under the DGP:
z1,z2,e1,e2,e3 ~ iid N(0,1)
x = z1 + z2 + e1 + e2
y = 1 + x + e1 + e3
Z = [z1 z2 junk], where junk are N(0,1) and do not enter x in DGP.
"""
function simulate_one_dgp(n, K)
    
    z1 = randn(n)
    z2 = randn(n)
    e1 = randn(n)
    e2 = randn(n)
    e3 = randn(n)

    x = z1 .+ z2 .+ e1 .+ e2
    y = 1 .+ x .+ e1 .+ e3
    
    if K > 2
        z_junk = randn(n, K-2)
        Z = hcat(z1, z2, z_junk)
    end

    if K == 2 
        Z = hcat(z1, z2)
    end

    return y, x, Z
end

# ------------------------------------------------------------
# Run simulations for all (n,K), store distributions
# ------------------------------------------------------------
# We’ll store all betas in a dictionary keyed by (n,K)
results = Dict{Tuple{Int,Int}, Dict{Symbol, Vector{Float64}}}()

for n in n_vec, K in K_vec
    βols = ones(R)
    βiv  = ones(R)

    println("Working on setup n = $n and k = $K ...")
    for r in 1:R
        println("Working on replication $r ...")
        y, x, Z = simulate_one_dgp(n, K)
        βols[r], βiv[r] = ols_2sls_slopes(y, x, Z)
    end
    results[(n,K)] = Dict(:ols => βols, :iv => βiv)
end

# ------------------------------------------------------------
# Summaries: median and IQR for horizontal bar visualization
# ------------------------------------------------------------
# For each combination, compute median and IQR (q75-q25) for OLS and IV
labels = String[]
med_ols = Float64[]
iqr_ols = Float64[]
med_iv  = Float64[]
iqr_iv  = Float64[]

for n in n_vec, K in K_vec
    push!(labels, "n=$(n), K=$(K)")
    βols = results[(n,K)][:ols]
    βiv  = results[(n,K)][:iv]

    qols = quantile(βols, [0.25, 0.5, 0.75])
    qiv  = quantile(βiv,  [0.25, 0.5, 0.75])

    push!(med_ols, qols[2]); push!(iqr_ols, qols[3]-qols[1])
    push!(med_iv,  qiv[2]);  push!(iqr_iv,  qiv[3]-qiv[1])
end

xpos = collect(1:length(labels))

# Side-by-side positions
offset = 0.18
xpos_ols = xpos .- offset
xpos_iv  = xpos .+ offset

half_iqr_ols = iqr_ols ./ 2
half_iqr_iv  = iqr_iv  ./ 2

p = plot(
    size=(1300, 700),
    xlabel="(n, K)",
    ylabel="Estimated slope (β̂ on x)",
    xticks=(xpos, labels),
    xrotation=35,
    legend=:topright,
    framestyle=:box
)

# Bars (medians)
bar!(p, xpos_ols, med_ols;
    label="OLS median",
    alpha=0.65,
    bar_width=0.32,
    linecolor=:transparent,
    color=:blue
)
bar!(p, xpos_iv, med_iv;
    label="2SLS (many IVs) median",
    alpha=0.65,
    bar_width=0.32,
    linecolor=:transparent,
    color=:red
)

# Error bars (IQR/2)
scatter!(p, xpos_ols, med_ols;
    yerror=half_iqr_ols,
    marker=:none,
    linecolor=:blue,
    label=""
)
scatter!(p, xpos_iv, med_iv;
    yerror=half_iqr_iv,
    marker=:none,
    linecolor=:red,
    label=""
)

# True beta reference line
hline!(p, [1.0], linestyle=:dash, linecolor=:black, label="true β=1")

display(p)

mkpath("images")
savefig(p, "images/comparison_ols_iv.png")


