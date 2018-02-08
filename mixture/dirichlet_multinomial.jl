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
    return log(n_x) + lbeta.(alpha_0, n_x) - sum(log(x[mask])) -
            sum(lbeta.(x[mask], alpha[mask, :]), 1)
end

function logp_dirichlet_multinomial(x::SparseVector{Int64}, alpha::Array{Float64,1})
    alpha = reshape(alpha, (size(alpha)..., 1))
    return logp_dirichlet_multinomial(x, alpha)[1]
end

function logp_dirichlet_multinomial(x::AbstractArray{Int64,2}, alpha::Float64)
    """Compute the marginal probability of x given alpha, where X follows the
    Dirichlet-multinomial distribution of parameter alpha.

    # Arguments
    - `x::SparseMatrix{Int64,Int64}`: the N-by-D matrix of observation counts
        for each bin. Each row is an observation
    - `alpha::Array{Float64,1}`: a float for the symmetric Dirichlet

    # Returns
    A Float64 for the marginal log probability of x given alpha
    """
    n_x = sum(x, 2)
    non_zero_rows = findnz(n_x)[1]
    x = x[non_zero_rows, :]
    N, D = size(x)
    alpha_0 = D*alpha
    x = vec(x)
    mask = findnz(x)[1]
    return sum(log.(n_x)) + sum(lbeta.(alpha_0, n_x)) -
            sum(log.(x[mask])) - sum(lbeta.(x[mask], alpha))
end

function update_dirichlet_parameters!(x::SparseVector{Int64},
        p::Vector{Float64}, alpha::Array{Float64,2})
    # BLAS.gemm!('N', 'T', 1.0, X[n, :], qzn, 1.0, qtheta)
    mask = findnz(x)[1]
    alpha[mask, :] += x[mask] .* p'
end
