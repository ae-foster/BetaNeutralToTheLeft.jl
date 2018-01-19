using ProgressMeter

true_dataset = true
debug = false

if true_dataset # Data from source
    include("dataset.jl")
    filename = "./reviews.json"
    X = getDocumentTermMatrixFromReviewsJson(filename)
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
# qz[i, :] represents the log probability that observation i is attribuetd to cluster k
qz = -Inf*ones(Float64, N, K_max)
# qtheta[:, k] represent natural parameter of a Dirichlet for cluster k
qtheta = zeros(Float64, D, K_max)

# Prior (hyper)parameters
dir_prior_param = 0.01 * ones(Float64, D)
a = 0.1 # inter-arrival Geometric success parameter
alpha = 0.5 # Neutral to the left parameter

# Partial sums for qz^pr
S_n = ones(Float64, K_max) # for efficiently computing E[n_k]
# Method for estimating qz
qzn_estimator_method = 1
# Only updated when using qzn_estimator_method == 2
Sprod = ones(Float64, K_max) # for efficiently computing E[K_{n-1}]

# Threshold for new cluster
epsilon = 0.1

# Initialization <=> First iteration
p = Progress(N, .5, "Observation n°: ", 50)
qz[1, 1] = 0 # Initialize first data point
qtheta[:, 1] = dir_prior_param + X[1, :] # Initialize qtheta posterior given x_1

function logp_dirichlet_multinomial(x::Union{Vector,SparseVector}, alpha::Vector)
    # cf https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    n_x = sum(x)
    alpha_0 = sum(alpha)
    nz = [(x_k, alpha_k) for (x_k, alpha_k) in zip(x, alpha) if x_k > 0]
    return log(n_x) + lbeta(alpha_0, n_x) - sum([log(x_k) + lbeta.(x_k, alpha_k) for (x_k, alpha_k) in nz])
end

function qzn_pr_estimator_1(S_n::Vector, n::Int, K_max::Int, alpha::Float64)
    unnormalized = max.((S_n - alpha),0)
    lp = log.(unnormalized ./ sum(unnormalized))
    return lp
end

function qzn_pr_estimator_2(S_n::Vector, Sprod::Vector, n::Int, K_max::Int, a::Float64, alpha::Float64)
    (1 - a) * max((S_n - alpha),0) / (n - 1 - alpha*(K_max - sum(Sprod)))
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
    tic()
    ProgressMeter.update!(p, n)
    print_debug("n: ", n)
    print_debug("K_max: ", K_max)
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
    if qzn_estimator_method == 1
        qzn_pr = log(1-a) + qzn_pr_estimator_1(S_n, n, K_max, alpha)
    elseif qzn_estimator_method != 1
        error("Currently unsupported")
    end
    qzn_pr_new = log(a)

    print_debug("qzn_pr ", qzn_pr)
    print_debug("qzn_pr_new ", qzn_pr_new)
    q_pr_time += toc()
    tic()

    # Should have
    # sum(exp(qzn_pr)) + exp(qzn_pr_new) = 1

    # Second projection: Use marginalization of conjugate exp fam
    # i.e. substraction in natural parameter space
    for k = 1:K_max
        qz[n,k] = qzn_pr[k] + logp_dirichlet_multinomial(X[n, :], qtheta[:, k])
        if isnan(qz[n,k])
            print_debug("That was a NaN")
            print_debug(X[n, :])
            print_debug(qtheta[:, k])
            error()
        end
    end
    qzn_new = qzn_pr_new + logp_dirichlet_multinomial(X[n, :], dir_prior_param)

    # Multiply this unnormalized distribution by a constant
    offset = max.(qz[n, :], qzn_new)[1]
    qz[n,:] -= offset
    qzn_new -= offset

    print_debug("qz[n,:] ", qz[n,:])
    print_debug("qzn_new", qzn_new)

    log_new_cluster_prob = qzn_new - log(exp(qzn_new) + sum(exp.(qz[n, :])))
    dir_time += toc()

    # Should create a new cluster ?
    print_debug("log_new_cluster_prob: ", log_new_cluster_prob)
    if log_new_cluster_prob > log(epsilon)
        tic()
        println("New cluster")
        K_max += 1
        push!(S_n, 0)
        if qzn_estimator_method == 2
            push!(Sprod, 1)
        end
        qtheta = hcat(qtheta, dir_prior_param)
        qz = hcat(qz, -Inf*ones(Float64, N, 1))
        qz[n, K_max] = qzn_new
        new_cluster_time += toc()
    end
    tic()

    print_debug("Unnormalized ", qz[n, :])
    qz[n, 1:K_max] -= log(sum(exp.(qz[n, 1:K_max]))) # normalization
    print_debug("Normalized ", qz[n, :])

    # Update global parameter approximation qtheta
    S_n += exp.(qz[n, 1:K_max]) # sufficient stats for expectation of n_k

    Scdf = 0
    for k = 1:K_max
        qtheta[:, k] += exp.(qz[n, k]) .* X[n, :] # update global parameter # Note: wrong ??
        if qzn_estimator_method == 2
            # Sufficient stats for expectation of Kn
            Scdf += qz[n, k]
            Sprod[k] *= Scdf
        end
    end
    qtheta_time += toc()

end
# print_debug(qz)
println("N: ", N)
println("D: ", D)
println("K_max: ", K_max)
println("qz: ", exp.(qz[1:10,1:10]))
if !true_dataset println("qz: ", qz) end


println("q_pr ", q_pr_time)
println("dir", dir_time)
println("new cluster", new_cluster_time)
println("qtheta", qtheta_time)
