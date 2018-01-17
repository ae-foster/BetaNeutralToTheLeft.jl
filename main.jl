using ProgressMeter

true_dataset = true

if true_dataset # Data from source
include("dataset.jl")
# Download http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz
filename = "/Users/EmileMathieu/code/NTL/reviews.json"
# Base.run(`sed '1s/^/[/;$!s/$/,/;$s/$/]/' reviews_Musical_Instruments_5.json > reviews.json`)
X = getDocumentTermMatrixFromReviewsJson(filename)
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

# Variational distributions: Mean field approximation
K_max = 1
qz = zeros(Float64, N, K_max) # Categorical/Discrete
qtheta = zeros(Float64, D, K_max) # natural parameter of a Dirichlet

# Prior (hyper)parameters
dir_prior_param = ones(Float64, D)
a = 0.1 # inter-arrival Geometric success parameter
alpha = 0.5 # Neutral to the left parameter

# Partial sums for qz^pr
S_n = ones(Float64, K_max) # for efficiently computing E[n_k]
Sprod = ones(Float64, K_max) # for efficiently computing E[K_{n-1}]

# Threshold for new cluster
epsilon = 0.1

# Initialization <=> First iteration
p = Progress(N, .5, "Observation n°: ", 50)
qz[1, 1] = 1 # Initialize first data point
qtheta[:, 1] = dir_prior_param + X[1, :] # Initialize qtheta posterior given x_1

function p_dirichlet_multinomial(x::Union{Vector,SparseVector}, alpha::Vector)
    # cf https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    n_x = sum(x)
    alpha_0 = sum(alpha)
    return factorial(n_x)*gamma(alpha_0)/gamma(n_x+alpha_0)*prod([gamma(x_k+alpha_k)/factorial(x_k)/gamma(alpha_k) for (x_k, alpha_k) in zip(x, alpha)])
end

function logp_dirichlet_multinomial(x::Union{Vector,SparseVector}, alpha::Vector)
    # cf https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution
    n_x = sum(x)
    alpha_0 = sum(alpha)
    return lfact(n_x)+lgamma(alpha_0)-lgamma(n_x+alpha_0)+sum([lgamma(x_k+alpha_k)-lfact(x_k)-lgamma(alpha_k) for (x_k, alpha_k) in zip(x, alpha)])
end

function qzn_pr_estimator_1(S_n::Vector, n::Int, K_max::Int, a::Float64, alpha::Float64)
    (1 - a) * max((S_n - alpha),0) / (n - 1 - alpha*K_max)
end

function qzn_pr_estimator_2(S_n::Vector, Sprod::Vector, n::Int, K_max::Int, a::Float64, alpha::Float64)
    (1 - a) * max((S_n - alpha),0) / (n - 1 - alpha*(K_max - sum(Sprod)))
end

for n = 2:N
    ProgressMeter.update!(p, n)
    println("n: ", n)
    if n > 10 break end

    ## Update local latent variable distributions qz_pr
    # First projection: Propagate (projection of predition term)
    # NOTE: function for different approximations (MC, 2nd term correction, etc)
    println("S_n: ", S_n)
    println("Sprod: ", Sprod)
    qzn_pr = qzn_pr_estimator_1(S_n, n, K_max, a, alpha)
    qzn_pr_new = a

    # Second projection: Use marginalization of conjugate exp fam
    # i.e. substraction in natural parameter space
    for k = 1:K_max
        qz[n,k] = qzn_pr[k] * exp(logp_dirichlet_multinomial(X[n, :], qtheta[:, k]))
    end
    qzn_new = qzn_pr_new * exp(logp_dirichlet_multinomial(X[n, :], dir_prior_param))

    # Should create a new cluster ?
    println("qz[n,:]: ", qz[n,:])
    println("qzn_new: ", qzn_new)
    println("qzn_new norm: ", qzn_new/(sum(qz[n,:])+qzn_new))
    if qzn_new/(sum(qz[n,1:K_max])+qzn_new) > epsilon
        println("New cluster")
        K_max += 1
        push!(S_n, 0)
        push!(Sprod, 1)
        qtheta = hcat(qtheta, dir_prior_param)
        qz = hcat(qz, zeros(Float64, N, 1))
        qz[n, K_max] = qzn_new
    end
    qz[n, 1:K_max] /= sum(qz[n, 1:K_max]) # normalization

    # Update global parameter approximation qtheta
    Scdf = 0
    S_n += qz[n, 1:K_max] # sufficient stats for expectation of n_k
    for k = 1:K_max
        qtheta[:, k] += qz[n, k] * X[n, :] # update global parameter # Note: wrong ??
        # Sufficient stats for expectation of Kn
        Scdf += qz[n, k]
        Sprod[k] *= Scdf
    end

end
# println(qz)
println("N: ", N)
println("D: ", D)
println("K_max: ", K_max)
println("qz: ", qz[1:10,:])
if !true_dataset println("qz: ", qz) end
