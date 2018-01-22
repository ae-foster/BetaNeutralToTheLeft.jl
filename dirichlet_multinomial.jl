################################################################################
# Utilities for Dirichlet-multinomial distribution
################################################################################

function logp_dirichlet_multinomial(x::SparseVector{Int64}, alpha::Array{Float64,2})
    """Compute the marginal probability of x given alpha, where X follows the
    Dirichlet-multinomial distribution of parameter alpha.

    # Arguments
    - `x::SparseVector{Int64}`: the D-vector of observation counts for each bin
    - `alpha::Array{Float64,2}`: a D-by-K matrix of possible parameters.
        Each column of alpha should represent a possible Dirichlet parameter

    # Returns
    A 1-by-K vector of marginal log probabilities
    """
    n_x = sum(x)
    alpha_0 = sum(alpha, 1)
    mask = findnz(x)[1]
    return log(n_x) + lbeta.(alpha_0, n_x) - sum(log(x[mask])) - sum(lbeta.(x[mask], alpha[mask, :]), 1)
end

function logp_dirichlet_multinomial(x::SparseVector{Int64}, alpha::Array{Float64,1})
    """Compute the marginal probability of x given alpha, where X follows the
    Dirichlet-multinomial distribution of parameter alpha.

    # Arguments
    - `x::SparseVector{Int64}`: the D-vector of observation counts for each bin
    - `alpha::Array{Float64,1}`: a D-vector of the Dirichlet parameter

    # Returns
    A Float64 for the marginal log probability of x given alpha
    """
    n_x = sum(x)
    alpha_0 = sum(alpha)
    mask = findnz(x)[1]
    return log(n_x) + lbeta(alpha_0, n_x) - sum(log(x[mask])) - sum(lbeta.(x[mask], alpha[mask]), )
end
