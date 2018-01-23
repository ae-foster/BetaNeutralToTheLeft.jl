using ProgressMeter
include("dirichlet_multinomial.jl")
include("estimators.jl")
include("dataset.jl")

##############################################################################
# Debugging
##############################################################################

debug = false

function print_debug(args...)
    if debug == true
        println(args...)
    end
end

##############################################################################
# Data
##############################################################################

true_dataset = false

if true_dataset # Data from source
    filename = "./reviews.json"
    z, X = getDocumentTermMatrixFromReviewsJson(filename)
else # Synthetic data
    println("Synthesising data")
    z, X = generateDataset(1000, 1000, 30, .5, 0.5, 0.1*ones(1000))
    println("Done")
end
N, D = size(X)

############################################################################
# Variational distributions: Mean field approximation
############################################################################

# K_max = Number of clusters allocated
K_max = 1
# K_n = Expected number of occupied  clusters
Kn = 1
# qz[i, :] represents the log probability that observation i is attribuetd to cluster k
log_qz = -Inf*ones(Float64, N, K_max)
# qtheta[:, k] represent the parameter (usually called alpha) of a Dirichlet for cluster k
qtheta = zeros(Float64, D, K_max)

# Prior (hyper)parameters
# If using Amazon musical instruments, MLE for this hyperparameter is 1e-2
dir_prior_param = 1e-1 * ones(Float64, D)
# Beta prior on geometric parameter
a_prime_prior = 1; b_prime_prior = 1
 # Neutral to the left parameter
alpha = 0.5

# Threshold for new cluster
epsilon = max(alpha, 1e-5)
# Accumulate probability 'lost' to uninstantiated clusters
acc_loss = 0

############################################################################
# Implementation specific data structures
############################################################################

# Method for estimating qz
qzn_estimator_method = 1
# Partial sums for qz^pr
S_n = ones(Float64, K_max) # for efficiently computing E[n_k]
# Only updated when using qzn_estimator_method == 2
Sprod = ones(Float64, K_max) # for efficiently computing E[K_{n-1}]

# Only updated when using qzn_estimator_method == 3 # Monte Carlo
M = 500
nk_stats = zeros(Int, M, K_max)


############################################################################
# Timings
############################################################################
q_pr_time = 0
dir_n_time = 0
dir_new_time = 0
new_cluster_time = 0
s_n_time = 0
qtheta_time = 0

############################################################################
# Iteration n=1
############################################################################

# Progress bar
p = Progress(N, .5, "Observation n°: ", 50)

# Put first data point in the first cluster
log_qz[1, 1] = 0 # Initialize first data point
qtheta[:, 1] = dir_prior_param + X[1, :] # Initialize qtheta posterior given x_1

#############################################################################
# Iteration n>1
#############################################################################

for n = 2:N
    # Move progress bar
    ProgressMeter.update!(p, n)

    # Debug info, timing
    print_debug("n: ", n)
    print_debug("K_max: ", K_max)
    print_debug("Kn: ", Kn)
    tic()

    # Skip empty documents: they cause NaNs
    if sum(X[n, :]) == 0
        continue
    end

    # First projection
    # Compute prior qzn_pr using selected method
    if qzn_estimator_method == 1
        log_qzn_pr, log_qzn_pr_new = qzn_pr_estimator_1(
            n, Kn, alpha, a_prime_prior, b_prime_prior, S_n)
    elseif qzn_estimator_method == 3
        nk_stats,log_qzn_pr,log_qzn_pr_new = qzn_pr_estimator_MC(
            exp.(log_qz[n-1,1:K_max]), nk_stats, M, n, K_max, alpha,
            a_prime_prior, b_prime_prior)
    else
        error("Currently unsupported")
    end

    # Debug info, timing
    print_debug("log_qzn_pr ", log_qzn_pr)
    print_debug("log_qzn_pr_new ", log_qzn_pr_new)
    q_pr_time += toq()
    tic()

    # Second projection: Use marginalization of conjugate exp fam
    # i.e. substraction in parameter space
    log_qz[n,:] = log_qzn_pr + logp_dirichlet_multinomial(X[n, :], qtheta)'

    # Check for NaN in log_qz[n, :]
    if any(isnan, log_qz[n,:])
        print_debug(X[n, :])X[n, :] .* qzn'
        print_debug(qtheta)
        error("NaN encountered in logp_dirichlet computation")
    end

    # Timings
    dir_n_time += toq()
    tic()

    # New cluster probability
    log_qzn_new = log_qzn_pr_new + logp_dirichlet_multinomial(
        X[n, :], dir_prior_param)

    # Debug info, timing
    print_debug("log_qz[n,:] ", log_qz[n,:])
    print_debug("log_qzn_new ", log_qzn_new)
    dir_new_time += toq()
    tic()

    # Use log-sum-exp trick to normalize
    # Initially, we only need the normalized value for the new cluster
    offset = max.(log_qz[n, :], log_qzn_new)[1]
    log_qz[n,:] -= offset
    log_qzn_new -= offset
    log_new_cluster_prob = log_qzn_new -
        log(exp(log_qzn_new) + sum(exp.(log_qz[n, :])))

    # Should create a new cluster ?
    print_debug("log_new_cluster_prob: ", log_new_cluster_prob)

    if log_new_cluster_prob > log(epsilon)
        # Debug info
        print_debug("New cluster")

        # Increase the size of arrays to hold new cluster
        K_max += 1
        qtheta = hcat(qtheta, dir_prior_param)
        log_qz = hcat(log_qz, -Inf*ones(Float64, N, 1))
        log_qz[n, K_max] = log_qzn_new

        # Update estimator-specific data structures
        if qzn_estimator_method < 3 push!(S_n, 0) end
        if qzn_estimator_method == 2 push!(Sprod, 1) end
        if qzn_estimator_method == 3
            nk_stats = hcat(nk_stats, zeros(Float64, M))
        end
    else
        # Accumulate the 'lost' probability
        acc_loss += exp(log_new_cluster_prob)
    end

    # Debug info, timings
    new_cluster_time += toq()
    tic()
    print_debug("Unnormalized ", log_qz[n, :])

    # normalization
    log_qz[n, 1:K_max] -= log(sum(exp.(log_qz[n, 1:K_max])))
    qzn = exp.(log_qz[n, 1:K_max])

    # sufficient stats for expectation of n_k
    if qzn_estimator_method < 3
        S_n += qzn
    end

    # Update Kn using probabilities
    if qzn_estimator_method == 1
        Kn = acc_loss + sum(min.(S_n, 1))
    end

    if qzn_estimator_method == 2
        Scdf = 0
        for k = 1:K_max
            # Sufficient stats for expectation of Kn
            Scdf += exp(log_qz[n, k])
            Sprod[k] *= Scdf
        end
    end

    # Debug info, timings
    print_debug("Normalized ", log_qz[n, :])
    s_n_time += toq()
    tic()


    # Update qtheta approximately using current assignment probabilities
    # BLAS.gemm!('N', 'T', 1.0, X[n, :], qzn, 1.0, qtheta)
    mask = findnz(X[n, :])[1]
    qtheta[mask, :] += X[n, mask] .* qzn'

    # Update the timer
    qtheta_time += toq()

end

###########################################################################
# Display results, diagnostics
###########################################################################

println("N: ", N)
println("D: ", D)
println("K_max: ", K_max)
println("Kn: ", Kn)

println("--------- computation times ----------")
println("q_pr ", q_pr_time)
println("dir_n ", dir_n_time)
println("dir_new ", dir_new_time)
println("new cluster ", new_cluster_time)
println("s_n ", s_n_time)
println("qtheta ", qtheta_time)

# Compute metrics of the approximate posterior distribution
include("metrics.jl")
qz = exp.(log_qz)
true_nb_clusters, approx_nb_cluster = compare_nb_clusters(z, qz)
println("true_no_clusters ", true_nb_clusters)
println("Cluster quality metric ", compute_clustering_quality(z, qz))
