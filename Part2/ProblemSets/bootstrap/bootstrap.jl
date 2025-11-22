using LinearAlgebra, Random, Distributions, Plots

cd("/Users/junbiao/Dropbox/PhD_firstyear/Metrics_I/EconGA_2001_Econometrics/Part2/ProblemSets/bootstrap")

# helper functions 
function ls_estimator(Y, X)
    n, k = size(X)
    @assert length(Y) == n
    β̂ = inv(X' * X) * (X' * Y)
    ê = Y .- X * β̂
    var_β̂ = ((ê' * ê) /(n - k)) * inv(X' * X)
    return β̂, var_β̂, ê
end

# t-ratio computation
function t_ratio(β̂, β_true, β̂_std)
    return (β̂ - β_true) / β̂_std
end

# true parameters
β0 = 1
β1 = 1

# Asymptotic 95th percentile for t-statistic
asymptotic_95 = quantile(Normal(0, 1), 0.95)

# Residual bootstrap (part a, b, c)
for n in [10, 50, 200]
    Random.seed!(n) 
    x = rand(Uniform(0,2), n)

    Random.seed!(n+1)
    e = rand(Uniform(-1,1), n)
    y = β0 .+ β1 .* x .+ e

    S = 202  # Number of bootstrap replications
    x_cat = hcat(ones(n), x)
    β̂, var_β̂, ê = ls_estimator(y, x_cat)

    # Bootstrap distributions
    β̂1_bs = zeros(S)
    t_stat_bs = zeros(S)

    for s in 1:S
        Random.seed!(s)
        e_sim = rand(ê , n)
        y_sim = x_cat * β̂ .+ e_sim
        β̂_bs, var_β̂_bs, _ = ls_estimator(y_sim, x_cat)
        β̂1_bs[s] = β̂_bs[2]
        β̂1_std_bs = sqrt(var_β̂_bs[2, 2])
        t_stat_bs[s] = t_ratio(β̂_bs[2], β1, β̂1_std_bs)
    end

    t_stat_bs = sort(t_stat_bs)[2:end-1]
    β̂1_bs = sort(β̂1_bs)[2:end-1]

    # Compute 95th percentiles of bootstrap distributions
    β̂1_95_bs = quantile(β̂1_bs, 0.95)
    t_stat_95_bs = quantile(t_stat_bs, 0.95)

    # Visualization
    p1 = histogram(β̂1_bs, xlims=(0, 2), label="Bootstrap β̂1", color=:blue, alpha=0.6)
    vline!(p1, [β̂1_95_bs], label="95th Percentile", color=:red, linestyle=:dash, linewidth=3)
    annotate!(p1, [(β̂1_95_bs, 3, text(string(round(β̂1_95_bs, digits=3)), :black, 12, :bold))])

    p2 = histogram(t_stat_bs, xlims=(0, 4),label="Bootstrap t-stat", color=:green, alpha=0.6)
    vline!(p2, [t_stat_95_bs], label="95th Percentile", color=:red, linestyle=:dash, linewidth=3)
    annotate!(p2, [(t_stat_95_bs, 3, text(string(round(t_stat_95_bs, digits=3)), :black, 12, :bold))])

    plot(p1, p2,
        layout=(1, 2),
        xlabel="Value",
        ylabel="Frequency",
        size=(800, 300)
    )
    println("Residual Bootstrap (n = $n):")
    println("  95th percentile of β̂1: $β̂1_95_bs")
    println("  95th percentile of t-stat: $t_stat_95_bs")
    println("  Asymptotic 95th percentile of t-stat: $asymptotic_95\n")
    savefig("images/bootstrap_n$n.png")
end

# question (d): Simulation using conditional distribution of residuals 

for n in [10, 50, 200]
    Random.seed!(n)
    x = rand(Uniform(0,2), n)

    S = 202  # number of replications
    x_cat = hcat(ones(n), x)
    β̂1_cond = zeros(S)
    t_stat_cond = zeros(S)

    for s in 1:S
        Random.seed!(s)
        e = rand(Uniform(-1,1), n)  # Generate new residuals from conditional distribution
        y = β0 .+ β1 .* x .+ e
        β̂_cond, var_β̂_cond, _ = ls_estimator(y, x_cat)
        β̂1_cond[s] = β̂_cond[2]
        β̂1_std_cond = sqrt(var_β̂_cond[2, 2])
        t_stat_cond[s] = t_ratio(β̂_cond[2], β1, β̂1_std_cond)
    end

    t_stat_cond = sort(t_stat_cond)[2:end-1]
    β̂1_cond = sort(β̂1_cond)[2:end-1]

    # Compute 95th percentiles of conditional distributions
    β̂1_95_cond = quantile(β̂1_cond, 0.95)
    t_stat_95_cond = quantile(t_stat_cond, 0.95)

    # Visualization
    p1 = histogram(β̂1_cond, xlims=(0, 2), label="Cond-Residual β̂1", color=:blue, alpha=0.6)
    vline!(p1, [β̂1_95_cond], label="95th Percentile", color=:red, linestyle=:dash, linewidth=3)
    annotate!(p1, [(β̂1_95_cond, 3, text(string(round(β̂1_95_cond, digits=3)), :black, 12, :bold))])

    p2 = histogram(t_stat_cond, bins = 10, xlims=(0, 4), label="Cond-Residual t-stat", color=:green, alpha=0.6)
    vline!(p2, [t_stat_95_cond], label="95th Percentile", color=:red, linestyle=:dash, linewidth=3)
    annotate!(p2, [(t_stat_95_cond, 3, text(string(round(t_stat_95_cond, digits=3)), :black, 12, :bold))])

    plot(p1, p2,
        layout=(1, 2),
        xlabel="Value",
        ylabel="Frequency",
        size=(800, 300)
    )
    println("Conditional Residuals (n = $n):")
    println("  95th percentile of β̂1: $β̂1_95_cond")
    println("  95th percentile of t-stat: $t_stat_95_cond")
    println("  Asymptotic 95th percentile of t-stat: $asymptotic_95\n")
    savefig("images/condition_dist_based_n$n.png")
end