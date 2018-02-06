using Distributions
using SpecialFunctions

function logjoint_derivative_Tj(x::Real, j::Int, Kn::Int, alpha::Real, p::Distributions.Beta{Float64}, nj_bar::Array{Int})
    # Compute the derivative of the log joint with respect to Tj

    grad = digamma(x - j * alpha) # First terms of the product
    grad += - digamma(x - 1 - (j - 1) * alpha)
    # grad += (j - 2) / (x - 1) - mean(p) # term from the Gamma approximation
    grad += - mean(p) # term from the Exponential approximation

    grad += -digamma(nj_bar[j] - x + 1) # terms from combinatorial coefficient
    grad += digamma(nj_bar[j-1] - x + 2)

    if j == Kn
        grad += mean(p)
        grad = min(grad, 0)
    end # Censoring term for last T_Kn

    grad
end

function logjoint_second_derivative_Tj(x::Real, j::Int, Kn::Int, alpha::Real, nj_bar::Array{Int})
    # Compute the 2nd derivative of the log joint with respect to Tj

    hess = trigamma(x - j * alpha) # First terms of the product
    hess += - trigamma(x - 1 - (j - 1) * alpha)

    hess += trigamma(nj_bar[j] - x + 1) # terms from combinatorial coefficient
    hess += -trigamma(nj_bar[j-1] - x + 2)

    hess
end


# function logjoint_derivative_alpha(x::Real, Tj::Array{Real}, n::Int, Kn::Int, nj::Array{Int})
function logjoint_derivative_alpha(x::Real, Tj, n::Int, Kn::Int, nj::Array{Int})
    # Compute the derivative of the log joint with respect to alpha

    grad = Kn * digamma(n - Kn * x) # First term of the joint

    vec = zeros(Float64, Kn)
    for j in 1:Kn
        vec[j] = -j*digamma(Tj[j] - j * x) - digamma(nj[j] - x)
        if j > 1 vec[j] += (j-1)*digamma(Tj[j] - 1 - (j - 1) * x) end
        vec[j] += digamma(1 - x)
    end
    grad += sum(vec)

    grad
end

function logjoint_second_derivative_alpha(x::Real, Tj, n::Int, Kn::Int, nj::Array{Int})
    # Compute the 2nd derivative of the log joint with respect to alpha
    hess = - Kn^2 * trigamma(n - Kn * x) # First term of the joint

    vec = zeros(Float64, Kn)
    for j in 1:Kn
        vec[j] = j^2*trigamma(Tj[j] - j * x) + trigamma(nj[j] - x)
        if j > 1 vec[j] += (j-1)^2*trigamma(Tj[j] - 1 - (j - 1) * x) end
        vec[j] += - trigamma(1 - x)
    end
    hess += sum(vec)

    hess
end


# Load dataset
include("dataset.jl")
nj, T_data = parseSnapData("data/sorted-sx-mathoverflow.txt")
Kn = length(nj)
println("Kn:",Kn)
n = sum(nj)
println("n:",n)
nj_bar = cumsum(nj)

# Posterior distribution of inter-arrival time Geomtric's parameter
ap = 1; bp = 1
p = Beta(ap + Kn - 1, bp + n - 1)

# Initialise point estimates
alpha = 0.
Tj = round.(cumsum((n / Kn) * ones(Real, Kn)))

# Optimisation hyperparameters
nb_epochs = 10
lr_Tj = .01
lr_alpha = .01

for i in 1:nb_epochs
    println(i)

    for j in 2:Kn # step for Tj, j=2,...,Kn
        # old = Tj[j]
        grad = logjoint_derivative_Tj(Tj[j], j, Kn, alpha, p, nj_bar)
        second_der = logjoint_second_derivative_Tj(Tj[j], j, Kn, alpha, nj_bar)
        Tj[j] += lr_Tj * (grad / second_der)

        if Tj[j] < j Tj[j] = j end
        if Tj[j] > nj_bar[j-1]+1 Tj[j] = nj_bar[j-1]+1 end
    end

    # step for alpha
    grad = logjoint_derivative_alpha(alpha, Tj, n, Kn, nj)
    second_der = logjoint_second_derivative_alpha(alpha, Tj, n, Kn, nj)
    alpha += lr_alpha * (grad / second_der)

    # TODO: compute & print log joint
end
