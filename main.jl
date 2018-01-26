using ProgressMeter
include("estimators.jl")
include("dataset.jl")
include("metrics.jl")

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
    z, X = generateGaussianDataset(250, 2, .5, 0.0, 100.0, 1.0)
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

# Beta prior on geometric parameter
a_prime_prior = 1; b_prime_prior = 1
 # Neutral to the left parameter
alpha = 0.0

# Threshold for new cluster
epsilon = max(alpha, 1e-5)
# Accumulate probability 'lost' to uninstantiated clusters
acc_loss = 0
# Track predictive log-likelihood
predictive_loglikelihood = zeros(Float64, N-1)

###########################################################################
# Emission specific parameters
###########################################################################
emission = "gaussian"

if emission == "dirichlet"
    include("dirichlet_multinomial.jl")
    # Function calculating logp
    logp_emission = logp_dirichlet_multinomial
    # Function updating parameters
    update_params! = update_dirichlet_parameters!

    # qtheta[1][:, k] represent the parameter (usually called alpha) of a Dirichlet for cluster k
    qtheta = (1e-2 * ones(Float64, D, K_max), )

    # Prior (hyper)parameters
    # If using Amazon musical instruments, MLE for this hyperparameter is 1e-2
    theta_prior = (1e-2 * ones(Float64, D), )

    # Control parameters: only relevant for Gaussian
    control_params = Tuple([])
elseif emission == "gaussian"
    include("gaussian.jl")
    logp_emission = logp_gaussian
    update_params! = update_gaussian_parameters!

    qtheta = (zeros(Float64, D, K_max), 1e-2 * ones(Float64, K_max))
    theta_prior = (zeros(Float64, D), 1e-2 * ones(Float64, K_max))
    control_params = (1.0, )
end


function adjoin(base::Tuple, addition::Tuple)
    return Tuple(if (ndims(b) > 1) hcat(b, a) else vcat(b, a) end
                 for (b, a) in zip(base, addition))
end

############################################################################
# Implementation specific data structures
############################################################################

# Method for estimating qz
qzn_estimator_method = 1
# Partial sums for qz^pr
S_n = ones(Float64, K_max) # for efficiently computing E[n_k]
# Only updated when using qzn_estimator_method == 2 # NRM
Sprod = ones(Float64, K_max-1) # for efficiently computing E[K_{n-1}]
Un_hat = 1

# Only updated when using qzn_estimator_method == 3 # Monte Carlo
M = 500
nk_stats = zeros(Int, M, K_max)


############################################################################
# Timings
############################################################################
q_pr_time = 0
logp_time = 0
new_cluster_time = 0
s_n_time = 0
qtheta_time = 0

############################################################################
# Iteration n=1
############################################################################

# Progress bar
p = Progress(N, .5, "Observation n°: ", 50)

# Put first data point in the first cluster
log_qz[1, 1] = 0 # Initialize probability
# Initialize qtheta posterior given x_1
update_params!(X[1, :], exp.(log_qz[1, :]), qtheta..., control_params...)

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
    elseif qzn_estimator_method == 2
        log_qzn_pr, log_qzn_pr_new, Un_hat = qzn_pr_estimator_NRM(S_n, Sprod, Un_hat, n, K_max, 1, 1000, 0.5)
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
    llikelihood = logp_emission(X[n, :], qtheta...)
    llikelihood_new = logp_emission(X[n, :], theta_prior...)

    log_qz[n,:] = log_qzn_pr + llikelihood[1:K_max]
    log_qzn_new = log_qzn_pr_new + llikelihood_new

    # Check for NaN in log_qz[n, :]
    if any(isnan, log_qz[n,:])
        print_debug(X[n, :])
        print_debug(qtheta)
        error("NaN encountered in logp_dirichlet computation")
    end

    # Debug, timing
    print_debug("log_qz[n,:] ", log_qz[n,:])
    print_debug("log_qzn_new ", log_qzn_new)
    logp_time += toq()
    tic()

    # Use log-sum-exp trick to normalize
    # Initially, we only need the normalized value for the new cluster
    offset = max.(log_qz[n, :], log_qzn_new)[1]
    log_qz[n,:] -= offset
    log_qzn_new -= offset
    normalizer = log(exp(log_qzn_new) + sum(exp.(log_qz[n, :])))
    log_new_cluster_prob = log_qzn_new - normalizer

    if predictive_loglikelihood != nothing
        predictive_loglikelihood[n-1] = offset + normalizer
    end

    # Should create a new cluster ?
    print_debug("log_new_cluster_prob: ", log_new_cluster_prob)

    if log_new_cluster_prob > log(epsilon)
        # Debug info
        print_debug("New cluster")

        # Increase the size of arrays to hold new cluster
        K_max += 1
        qtheta = adjoin(qtheta, theta_prior)
        log_qz = hcat(log_qz, -Inf*ones(Float64, N, 1))
        log_qz[n, K_max] = log_qzn_new

        # Update estimator-specific data structures
        if qzn_estimator_method < 3 push!(S_n, 0) end
        if qzn_estimator_method < 3 push!(Sprod, 1) end
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
        Sprod = Sprod .* (1 - qzn[2:K_max])
        Kn = acc_loss + K_max - sum(Sprod)
        #Kn = acc_loss + sum(min.(S_n, 1))
    end

    # Debug info, timings
    print_debug("Normalized ", log_qz[n, :])
    s_n_time += toq()
    tic()

    # Update qtheta approximately using current assignment probabilities
    update_params!(X[n, :], qzn, qtheta..., control_params...)

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
println("logp ", logp_time)
println("new cluster ", new_cluster_time)
println("s_n ", s_n_time)
println("qtheta ", qtheta_time)

# Compute metrics of the approximate posterior distribution
qz = exp.(log_qz)
true_nb_clusters, approx_nb_cluster = compare_nb_clusters(z, qz)
println("true_no_clusters ", true_nb_clusters)
println("Cluster quality metric ", compute_clustering_quality(z, qz))

plot_heatmap = true
if plot_heatmap
    include("plot_utils.jl")
    z_qz_heatmap(z, qz)
end

plot_scatter = true
if plot_scatter
    include("plot_utils.jl")
    gaussian_scatters(X, z, qz)
end

plot_pll = true
if plot_pll
    include("plot_utils.jl")
    pll_plot(predictive_loglikelihood)
end
