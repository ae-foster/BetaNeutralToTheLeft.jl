###########################################################################
# Utilities for the Gaussian family
###########################################################################
# For unknown variance, use http://www.fil.ion.ucl.ac.uk/~wpenny/publications/bmn.pdf

function logp_gaussian(x::Vector{Float64}, m::Array{Float64,2},
        tau::Vector{Float64})
    """Compute the marginal probability of x given the Gaussian parameters
    m and sigma2

    # Arguments
    - `x::Vector{Float64}`: the D-vector observation
    - `m::Array{Float64,2}`: the D-by-K matrix of possible means
    - `tau::Vector{Float64}`: the K-vector of possible precisions

    # Returns
    A K-vector of the marginal probabilities of x given the various parameter
    settings
    """
    d, k = size(m)
    dots = [dot(m[:, j] - x, m[:, j] - x) for j=1:k]
    return (d/2)log.(tau) - (tau/2) .* dots
end

function logp_gaussian(x::Vector{Float64}, m::Array{Float64,1},
        tau::Array{Float64,1})
    m = reshape(m, (size(m)..., 1))
    return logp_gaussian(x, m, tau)[1]
end

function update_gaussian_parameters!(x::Vector{Float64}, p::Vector{Float64},
        m::Array{Float64,2}, tau::Vector{Float64}, observe_tau::Float64)
    outer = x .* p'
    d, k = size(m)
    for j=1:k
        m[:, j] = (m[:, j] * tau[j] + observe_tau * outer[:, j]) / (tau[j] + p[j]*observe_tau)
        tau[j] += observe_tau * p[j]
    end
end
