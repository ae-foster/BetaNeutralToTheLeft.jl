# Data from source
D = 4 # vocab size
N = 5 # data size
X = Matrix{Int32}(N,D)
X = reshape([1 1 0 0
             2 1 0 1
             0 0 2 1
             0 0 1 2
             1 2 1 0 ], N, D)

K_max = 1
qz = zeros(Float64, N, K_max)
qtheta = zeros(Float64, D, K_max)

# Prior
dir_prior_param = ones(Float64, D)
a = 0.1
alpha = 0.5

# Partial sums for qz^pr
S_nk = ones(Float64, K_max)
Sprod = ones(Float64, K_max)

# Threshold for new cluster
epsilon = 0

# Initialize first data point
qz[1, 1] = 1
# Initialize qtheta posterior given x_1
qtheta[:, 1] = dir_prior_param + X[1, :]

for n = 2:N
    println("n: ", n)
    # Compute qz_pr: Propagate (projection of predition term)
    # NOTE: function for different approximations (MC, 2nd term correction, etc)
    qzn_pr = (1 - a) * (S_nk - alpha) / (n - 1 - alpha*(K_max - sum(Sprod)))
    qzn_pr_new = a

    # Compute qz: using marginalization of conjugate exp fam, i.e. substraction
    for k = 1:K_max
        alpha0 = sum(qtheta[:, k])
        tot_words = sum(X[n, :])
        marginal = factorial(tot_words)*gamma(alpha0)/gamma(tot_words+alpha0)*prod([gamma(x+alph)/factorial(x)/gamma(alph) for (x, alph) in zip(X[n, :], qtheta[:, k])])
        qz[n,k] = qzn_pr[k] * marginal
    end
    alpha0 = D
    tot_words = sum(X[n, :])
    marginal = factorial(tot_words)*gamma(alpha0)/gamma(tot_words+alpha0)*prod([gamma(x+alph)/factorial(x)/gamma(alph) for (x, alph) in zip(X[n, :], dir_prior_param)])
    println("marginal: ", marginal)
    qzn_new = qzn_pr_new * marginal

    # Create or not new cluster
    println("qzn_new: ", qzn_new)
    if qzn_new > epsilon
        K_max += 1
        push!(S_nk, 0)
        push!(Sprod, 1)
        qtheta = hcat(qtheta, dir_prior_param)
        qz = hcat(qz, zeros(Float64, N, 1))
        qz[n, K_max] = qzn_new
    end
    qz[n, :] /= sum(qz[n, :])

    # Compute qtheta
    Scdf = 0
    for k = 1:K_max
        qtheta[:, k] += qz[n, k] * X[n, :] # update parameter
        S_nk += qz[n, k]
        # Sufficient stats for expectation of Kn
        Scdf += qz[n, k]
        Sprod[k] *= Scdf
    end

end
println(qz)
