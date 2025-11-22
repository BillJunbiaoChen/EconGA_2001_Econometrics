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


function t_ratio(β̂, β_true, β̂_std)
    t_ratio = (β̂ - β_true) / β̂_std
    return t_ratio
end


# true parameters
β0 = 1
β1 = 1

# question (a), (b), (c): Residual boostrap
for n in [10, 50, 200]
    k = 2
    Random.seed!(n)
    x = rand(Uniform(0,2), n)

    Random.seed!(n)
    e = rand(Uniform(-1,1), n)
    y = β0 .+ β1 .* x .+ e

    # once the data is simulate, 
    # use residual boostrap to obtain the distributions of β̂ and t-ratio

    S = 200 # number of replications
    x_cat = hcat(ones(n), x)
    β̂, var_β̂, ê = ls_estimator(y, x_cat)
    β̂1_bs = ones(S)
    t_stat_bs = ones(S)

    for s in 1:S 
        Random.seed!(s)
        e_sim = rand(ê , n)
        y_sim = x_cat * β̂ .+ e_sim 
        β̂_bs, var_β̂_bs, _ = ls_estimator(y_sim, x_cat)
        β̂1_bs[s] = β̂_bs[2]      
        β̂1_std_bs = sqrt(var_β̂_bs[2, 2])
        t_stat_bs[s] = t_ratio(β̂_bs[2], β0, β̂1_std_bs)
    end

    # visualization 
    plot(
        histogram(β̂1_bs, xlims=(1.5, 2.5), bins=10, label="LS (Residual Bootstrap)", color=:blue, alpha=0.5, linecolor=:transparent),
        histogram(t_stat_bs, bins=10, label="t-stat", color=:blue, alpha=0.5, linecolor=:transparent);
        layout=(1, 2),  # Arrange plots in a 1x2 grid
        size=(680, 300)  # Set the overall figure size
    )

    savefig("images/bootstrap_n$n.png")
end

# question (d): Simulation using conditional distribution of residuals 

for n in [10, 50, 200]
    k = 2
    Random.seed!(n)
    x = rand(Uniform(0,2), n)

    S = 200 # number of replications
    x_cat = hcat(ones(n), x)
    β̂1_bs = ones(S)
    t_stat_bs = ones(S)

    for s in 1:S 
        Random.seed!(s)
        e = rand(Uniform(-1,1), n)
        y = β0 .+ β1 .* x .+ e

        β̂_bs, var_β̂_bs, _ = ls_estimator(y, x_cat)
        
        β̂1_bs[s] = β̂_bs[2]      
        β̂1_std_bs = sqrt(var_β̂_bs[2, 2])
        t_stat_bs[s] = t_ratio(β̂_bs[2], β0, β̂1_std_bs)
    end

    # visualization 
    plot(
        histogram(β̂1_bs, bins=10, xlims=(0, 2), label="LS (cond-dist based)", color=:blue, alpha=0.5, linecolor=:transparent),
        histogram(t_stat_bs, bins=10, label="t-stat", color=:blue, alpha=0.5, linecolor=:transparent);
        layout=(1, 2),  # Arrange plots in a 1x2 grid
        size=(680, 300)  # Set the overall figure size
    )

    savefig("images/condition_dist_based_n$n.png")
end