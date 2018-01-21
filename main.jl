using ProgressMeter
using Distributions

true_dataset = true
debug = false

if true_dataset # Data from source
    include("dataset.jl")
    filename = "./reviews.json"
    z, X = getDocumentTermMatrixFromReviewsJson(filename)
    X = X[1:500, :]
    N, D = size(X)

else # Synthetic data
    D = 4 # vocab size
    N = 5 # data size
    X = Matrix{Int32}(N,D)
    X = reshape([1 1 0 0
                 2 1 0 1
                 0 0 2 1
                 0 0 1 2
                 1 2 1 0 ], N, D)
end

#######################################################
## Variational distributions: Mean field approximation
#######################################################

# K_max = Number of clusters allocated
K_max = 1
K_max_alloc = 200 # preallocate matrices
# qz[i, :] represents the log probability that observation i is attribuetd to cluster k
log_qz = -Inf*ones(Float64, N, K_max)
# qtheta[:, k] represent the parameter (usually called alpha) of a Dirichlet for cluster k
qtheta = zeros(Float64, D, K_max)

# Prior (hyper)parameters
dir_prior_param = 0.01 * ones(Float64, D)
a = 0.1 # inter-arrival Geometric success parameter
a_prime = 1; b_prime = 1; # Beta prior on a
alpha = 0.5 # Neutral to the left parameter

# Partial sums for qz^pr
S_n = ones(Float64, K_max) # for efficiently computing E[n_k]
# Method for estimating qz
qzn_estimator_method = 3
# Only updated when using qzn_estimator_method == 2
Sprod = ones(Float64, K_max) # for efficiently computing E[K_{n-1}]

# Only updated when using qzn_estimator_method == 3 # Monte Carlo
M = 50
nk_stats = zeros(Int, M, K_max)
T_Kprev = zeros(Int, M)

# Threshold for new cluster
epsilon = 0.1

# Initialization <=> First iteration
p = Progress(N, .5, "Observation n°: ", 50)
log_qz[1, 1] = 0 # Initialize first data point
qtheta[:, 1] = dir_prior_param + X[1, :] # Initialize qtheta posterior given x_1

function logp_dirichlet_multinomial(x::Union{Vector,SparseVector}, alpha::Vector)
    # cf https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    n_x = sum(x)
    alpha_0 = sum(alpha)
    mask = findnz(x)[1]
    return log(n_x) + lbeta(alpha_0, n_x) - sum(log(x[mask])) + sum(lbeta.(x[mask], alpha[mask]))
end

function qzn_pr_estimator_1(S_n::Vector, n::Int, K_max::Int, alpha::Float64)
    unnormalized = max.((S_n - alpha),0)
    lp = log.(unnormalized ./ sum(unnormalized))
    return lp
end

function qzn_pr_estimator_2(S_n::Vector, Sprod::Vector, n::Int, K_max::Int, a::Float64, alpha::Float64)
    (1 - a) * max((S_n - alpha),0) / (n - 1 - alpha*(K_max - sum(Sprod)))
end

function qzn_pr_estimator_MC(qz_prev:: Vector, nk_stats::Matrix, T_Kprev::Vector, M::Int, n::Int, K_max::Int, a::Float64, alpha::Float64)
    # Monte Carlo estimate with M samples
    # nk_stats Matrix of size MxKmax with n_k
    # T_Kprev vector of size M with T_K_{n-1}

    qz_prev_samples = wsample(1:K_max, qz_prev, M) # z ~ \hat{q}_{n-1}(.)
    # update nk
    for m in 1:M
        if nk_stats[m, qz_prev_samples[m]] == 0 T_Kprev[m] = n-1 end
        nk_stats[m, qz_prev_samples[m]] += 1
    end
    Kprev = sum(nk_stats .!= 0, 2)
    qzn_pr = zeros(Float64, K_max + 1)
    p_new_cluster = (Kprev - 1 + a_prime) ./ (T_Kprev + a_prime + b_prime)
    qzn_pr[K_max+1] = mean(p_new_cluster)
    for k in 1:K_max
        qzn_pr[k] = mean( (1 - p_new_cluster) .* (nk_stats[:,k] - alpha) ./ (n - 1 - alpha.*Kprev) )
    end

    nk_stats, T_Kprev, log.(qzn_pr[1:K_max]),  log(qzn_pr[K_max+1])
end

function print_debug(args...)
    if debug == true
        println(args...)
    end
end

q_pr_time = 0
dir_time = 0
new_cluster_time = 0
qtheta_time = 0

for n = 2:N
    ProgressMeter.update!(p, n)
    print_debug("n: ", n)
    print_debug("K_max: ", K_max)

    # Exclude empty documents: they cause NaNs
    if sum(X[n, :]) == 0
        continue
    end

    ## Update local latent variable distributions qz_pr
    # First projection: Propagate (projection of predition term)
    # NOTE: function for different approximations (MC, 2nd term correction, etc)
    # print_debug("S_n: ", S_n)
    # if qzn_estimator_method == 2
    #     print_debug("Sprod: ", Sprod)
    # end
    tic()
    if qzn_estimator_method == 1
        log_qzn_pr = log(1-a) + qzn_pr_estimator_1(S_n, n, K_max, alpha)
        log_qzn_pr_new = log(a)
    elseif qzn_estimator_method == 3
        nk_stats,T_Kprev,log_qzn_pr,log_qzn_pr_new = qzn_pr_estimator_MC(exp.(log_qz[n-1,1:K_max]), nk_stats, T_Kprev, M, n, K_max, a, alpha)
    else
        error("Currently unsupported")
    end

    print_debug("log_qzn_pr ", log_qzn_pr)
    print_debug("log_qzn_pr_new ", log_qzn_pr_new)
    q_pr_time += toq()
    tic()

    # Should have
    # sum(exp(qzn_pr)) + exp(qzn_pr_new) = 1

    # Second projection: Use marginalization of conjugate exp fam
    # i.e. substraction in parameter space
    for k = 1:K_max
        log_qz[n,k] = log_qzn_pr[k] + logp_dirichlet_multinomial(X[n, :], qtheta[:, k])
        if isnan(log_qz[n,k])
            print_debug("That was a NaN")
            print_debug(X[n, :])
            print_debug(qtheta[:, k])
            error()
        end
    end
    log_qzn_new = log_qzn_pr_new + logp_dirichlet_multinomial(X[n, :], dir_prior_param)

    # Multiply this unnormalized distribution by a constant
    offset = max.(log_qz[n, :], log_qzn_new)[1]
    log_qz[n,:] -= offset
    log_qzn_new -= offset

    print_debug("log_qz[n,:] ", log_qz[n,:])
    print_debug("log_qzn_new", log_qzn_new)

    dir_time += toq()
    tic()
    log_new_cluster_prob = log_qzn_new - log(exp(log_qzn_new) + sum(exp.(log_qz[n, :])))

    # Should create a new cluster ?
    print_debug("log_new_cluster_prob: ", log_new_cluster_prob)
    if log_new_cluster_prob > log(epsilon)
        println("New cluster")
        K_max += 1
        qtheta = hcat(qtheta, dir_prior_param)
        log_qz = hcat(log_qz, -Inf*ones(Float64, N, 1))
        log_qz[n, K_max] = log_qzn_new
        if qzn_estimator_method < 3 push!(S_n, 0) end
        if qzn_estimator_method == 2 push!(Sprod, 1) end
        if qzn_estimator_method == 3 nk_stats = hcat(nk_stats, zeros(Float64, M)) end
    end
    new_cluster_time += toq()
    tic()

    print_debug("Unnormalized ", log_qz[n, :])
    log_qz[n, 1:K_max] -= log(sum(exp.(log_qz[n, 1:K_max]))) # normalization
    print_debug("Normalized ", log_qz[n, :])

    # sufficient stats for expectation of n_k
    if qzn_estimator_method < 3 S_n += exp.(log_qz[n, 1:K_max]) end

    Scdf = 0
    for k = 1:K_max
        # Update global parameter approximation qtheta
        qtheta[:, k] += exp.(log_qz[n, k]) .* X[n, :] # update global parameter
        if qzn_estimator_method == 2
            # Sufficient stats for expectation of Kn
            Scdf += exp(log_qz[n, k])
            Sprod[k] *= Scdf
        end
    end
    qtheta_time += toq()

end

qz = exp.(log_qz)

# print_debug(log_qz)
println("N: ", N)
println("D: ", D)
println("K_max: ", K_max)
println("qz: ", qz[1:10,1:10])
if !true_dataset println("qz: ", qz) end

println("--------- computation times ----------")
println("q_pr ", q_pr_time)
println("dir ", dir_time)
println("new cluster ", new_cluster_time)
println("qtheta ", qtheta_time)

# Compute metrics of the approximate posterior distribution
include("metrics.jl")
true_nb_clusters, approx_nb_cluster = compare_nb_clusters(z, qz)
compute_clustering_quality(z, qz)
