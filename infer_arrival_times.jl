using Distributions
using SpecialFunctions

function logjoint_derivative_Tj(x::Real, j::Int, alpha::Real, p::Distributions.Beta{Float64}, nj_bar::Array{Int})
    res = digamma(x - j * alpha)
    res += (j - 2) / (x - 1) - mean(p)
    res += -digamma(nj_bar[j] - x + 1)
    if j > 1
        res += - digamma(x - 1 - (j - 1) * alpha) + digamma(nj_bar[j-1] - x + 2)
    end
    return res
end

# function logjoint_derivative_alpha(x::Real, Tj::Array{Real}, n::Int, Kn::Int, nj_bar::Array{Int})
function logjoint_derivative_alpha(x::Real, Tj, n::Int, Kn::Int, nj_bar::Array{Int})
    vec = zeros(Float64, Kn)
    for j in 1:Kn
        vec[j] = -j*digamma(Tj[j] - j * alpha) - digamma(nj[j]) - digamma(nj[j] - x)
        vec[j] += (j-1)*digamma(Tj[j] - 1 - (j - 1) * x) + digamma(1 - x)
    end
    return Kn * digamma(n - Kn * x) + sum(vec)
end

# Load dataset
include("dataset.jl")
nj, T_data = parseSnapData("data/sorted-sx-mathoverflow.txt")
Kn = length(nj)
n = sum(nj)
nj_bar = cumsum(nj)

# Posterior distribution of inter-arrival time Geomtric's parameter
ap = 1; bp = 1
p = Beta(ap + Kn - 1, bp + n - 1)

# Initialise point estimates
alpha = -2.
Tj = cumsum(round(n / Kn) * ones(Real, n))

# Optimisation hyperparameters
nb_epochs = 1
lr_Tj = .1
lr_alpha = .01
println(mean(Tj))
for i in 1:nb_epochs
    println(i)
    for j in 1:Kn
        # step for Tj, j=1,...,Kn
        old = Tj[j]
        grad = logjoint_derivative_Tj(Tj[j], j, alpha, p, nj_bar)
        println("grad:", grad)
        Tj[j] += lr_Tj * logjoint_derivative_Tj(Tj[j], j, alpha, p, nj_bar)
        println("Tj[",j,"]=",old," | ",Tj[j])
        # if j > 10 break end
    end
    println(mean(Tj))
    # step for alpha
    alpha +=-lr_alpha * logjoint_derivative_alpha(alpha, Tj, n, Kn, nj_bar)

    # Compute & print log joint ?
end
