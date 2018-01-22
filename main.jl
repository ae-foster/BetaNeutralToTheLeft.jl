using ProgressMeter
using Distributions
include("dirichlet_multinomial.jl")
include("estimators.jl")

true_dataset = true
debug = false

if true_dataset # Data from source
    include("dataset.jl")
    filename = "./reviews.json"
    z, X = getDocumentTermMatrixFromReviewsJson(filename)
    X = X[1:10000, :]
    z = z[1:10000]
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
# MLE for this hyperparameter is 1e-2
dir_prior_param = 1e-2 * ones(Float64, D)
a_prime = 1; b_prime = 1; # Beta prior on geometric parameter
alpha = 0.5 # Neutral to the left parameter

# Partial sums for qz^pr
S_n = ones(Float64, K_max) # for efficiently computing E[n_k]
# Method for estimating qz
qzn_estimator_method = 1
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

function print_debug(args...)
    if debug == true
        println(args...)
    end
end

q_pr_time = 0
dir_n_time = 0
dir_new_time = 0
new_cluster_time = 0
norm_time = 0
s_n_time = 0
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
        a = a_prime / (a_prime + b_prime)
        log_qzn_pr = log(1-a) + qzn_pr_estimator_1(S_n, n, K_max, alpha)
        log_qzn_pr_new = log(a)
    elseif qzn_estimator_method == 3
        nk_stats,T_Kprev,log_qzn_pr,log_qzn_pr_new = qzn_pr_estimator_MC(exp.(log_qz[n-1,1:K_max]), nk_stats, T_Kprev, M, n, K_max, a, alpha, a_prime, b_prime)
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
    log_qz[n,:] = log_qzn_pr + logp_dirichlet_multinomial(X[n, :], qtheta)'
    if any(isnan, log_qz[n,:])
        print_debug(X[n, :])X[n, :] .* qzn'
        print_debug(qtheta)
        error("NaN encountered in logp_dirichlet computation")
    end

    dir_n_time += toq()
    tic()

    log_qzn_new = log_qzn_pr_new + logp_dirichlet_multinomial(X[n, :], dir_prior_param)

    print_debug("log_qz[n,:] ", log_qz[n,:])
    print_debug("log_qzn_new ", log_qzn_new)

    # Multiply this unnormalized distribution by a constant
    offset = max.(log_qz[n, :], log_qzn_new)[1]
    log_qz[n,:] -= offset
    log_qzn_new -= offset

    dir_new_time += toq()
    tic()
    log_new_cluster_prob = log_qzn_new - log(exp(log_qzn_new) + sum(exp.(log_qz[n, :])))

    # Should create a new cluster ?
    print_debug("log_new_cluster_prob: ", log_new_cluster_prob)
    if log_new_cluster_prob > log(epsilon)
        print_debug("New cluster")
        K_max += 1
        qtheta = hcat(qtheta, dir_prior_param)
        log_qz = hcat(log_qz, -Inf*ones(Float64, N, 1))
        log_qz[n, K_max] = log_qzn_new
        if qzn_estimator_method < 3 push!(S_n, 0) end
        if qzn_estimator_method == 2 push!(Sprod, 1) end
        if qzn_estimator_method == 3 nk_stats = hcat(nk_stats, zeros(Float64, M)) end
        a_prime += 1
    else
        b_prime += 1
    end
    new_cluster_time += toq()
    tic()

    print_debug("Unnormalized ", log_qz[n, :])

    log_qz[n, 1:K_max] -= log(sum(exp.(log_qz[n, 1:K_max]))) # normalization
    qzn = exp.(log_qz[n, 1:K_max])

    print_debug("Normalized ", log_qz[n, :])

    norm_time += toq()
    tic()

    # sufficient stats for expectation of n_k
    if qzn_estimator_method < 3
        S_n += qzn
    end

    s_n_time += toq()
    tic()

    # qtheta += X[n, :] .* qzn'
    # BLAS.gemm!('N', 'T', 1.0, X[n, :], qzn, 1.0, qtheta)
    mask = findnz(X[n, :])[1]
    qtheta[mask, :] += X[n, mask] .* qzn'

    if qzn_estimator_method == 2
        Scdf = 0
        for k = 1:K_max
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
println("a_prime: ", a_prime, ", b_prime: ", b_prime)
if !true_dataset println("qz: ", qz) end

println("--------- computation times ----------")
println("q_pr ", q_pr_time)
println("dir_n ", dir_n_time)
println("dir_new ", dir_new_time)
println("new cluster ", new_cluster_time)
println("norm ", norm_time)
println("s_n ", s_n_time)
println("qtheta ", qtheta_time)

# Compute metrics of the approximate posterior distribution
include("metrics.jl")
true_nb_clusters, approx_nb_cluster = compare_nb_clusters(z, qz)
println("true_no_clusters ", true_nb_clusters)
println("Cluster quality metric ", compute_clustering_quality(z, qz))
