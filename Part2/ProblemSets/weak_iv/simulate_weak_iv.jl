using LinearAlgebra, Random, Distributions, Plots, StatsBase

cd("/Users/junbiao/Dropbox/PhD_firstyear/Metrics_I/EconGA_2001_Econometrics/Part2/ProblemSets/weak_iv")



K_vec = [2, 10, 100, 200]
n_vec = [20, 200, 2000]

# simulate the model 
K = K_vec[1]
n = n_vec[1]

# (zi1, zi2, ei1, ei2, ei3) are i.i.d. standard normal.

z1 = rand(Normal(0, 1), n)
z2 = rand(Normal(0, 1), n)
e1 = rand(Normal(0, 1), n)
e2 = rand(Normal(0, 1), n)
e3 = rand(Normal(0, 1), n)

x = z1 .+ z2 .+ e1 .+ e2 
y = 1 .+ x .+ e1 .+ e3
z_junk = rand(Normal(0, 1), (n, K-2))

# OLS 
β̂_OLS = inv(x' * x) * (x' * y)

# 2SLS with many junk IVs 
Z = hcat(z1, z2, z_junk)
FS_x_fit = Z * inv(Z' * Z) * Z' * x
β̂_2SLS = inv(FS_x_fit' * FS_x_fit) * (FS_x_fit' * y)