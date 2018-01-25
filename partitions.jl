###########################################################################
# Utilities for partition updates
###########################################################################
using StatsBase

function logp_partition(PP::Vector{Int},T::Vector{Int},Psi::Vector{Float64},
        alpha::Float64,interarrival_dist::DiscreteDistribution,is_partition::Bool)
    """
    - `PP`: vector of partition block sizes ordered by arrival time
    - `T`: vector of arrival times
    - `Psi`: vector of beta random variables (can be log(Psi))
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `interarrival_dist`: distribution object corresponding to i.i.d. interarrivals
    - `is_partition`: flag for computing binomial coefficients
    """

    if all(Psi .<= 0)
      # warn("Psi in log space.")
      log_Psi = Psi
      log_Psi_c = log.(1 - exp.(log_Psi))
    elseif all(0 .<= Psi .<= 1)
      log_Psi = log.(Psi)
      log_Psi_c = log.(1 - Psi)
    else
      error("Invalid Ψ.")
    end
    # shift distributions with non-zero mass on zero
    zero_shift = Int(minimum(interarrival_dist) == 0)

    PP_partial = cumsum(PP)
    # pop!(PP_partial)
    ia = T[2:end] .- T[1:(end-1)]

    K = size(Psi,1)
    idx = 1:(K-1)
    N = sum(PP)

    log_p = dot((PP[2:end] .- alpha .- 1),log_Psi[2:end]) + dot((PP_partial[1:(end-1)] .- idx.*alpha .- 1), log_Psi_c[2:end])
    # include arrival times
    log_p += sum(logpdf(interarrival_dist, ia))
    N - T[end] > 0 ? log_p += log(1 - cdf(interarrival_dist, N-T[end]-zero_shift)) : nothing
    log_p += -sum([lbeta(1 - alpha,T[j] - 1 - (j-1)*alpha) for j in 2:K])
    # include binomial coefficients if for a partition
    if is_partition
      log_p += sum([logBinomial(PP_partial[j] - T[j],PP[j] - 1) for j in 2:K])
    end

    return log_p

end

function logBinomial(n::Int,k::Int)
    """
    compues log of binomial coefficient {n choose k}
    """
    ret = lgamma(n+1) - lgamma(k+1) - lgamma(n - k + 1)
    return ret
end


function tallyInts(Z::Vector{Int},K::Int)
    """
    counts occurrences in `Z` of integers 1 to `K`
    - `Z`: vector of integers
    - `K`: maximum value to count occurences in `Z`
    """
    ret = zeros(Int,K)
    for j in 1:K
      ret[j] = sum(Z .== j)
    end
    return ret
end


function seq2part(Z::Vector{Int})
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    """
    # transform Z into an ordered partition and get arrival times
    K = maximum(Z)
    PP = tallyInts(Z,K)

    T = zeros(Int,K)
    for j in 1:K
      T[j] = findfirst(Z.==j)
    end
    return PP,T
end


function logp_labelSequence(Z::Vector{Int},Psi::Vector{Float64},
        alpha::Float64,interarrival_dist::DiscreteDistribution)
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    - `Psi`: vector of beta random variables (can be log(Psi))
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `interarrival_dist`: distribution object corresponding to i.i.d. interarrivals
    """

    PP,T = seq2part(Z)
    log_p = logp_partition(PP,T,Psi,alpha,interarrival_dist,false)
    return log_p

end


function update_psi_parameters(PP::Vector{Int},alpha::Float64)
    """
    - `PP`: arrival-ordered vector of partition block sizes
    - `alpha`: 'discount' parameter
    """
    K = size(T,1)
    PP_partial = cumsum(PP)

    Psi = zeros(Float64,K)
    Psi[1] = 1
    for j in 2:K
      Psi[j] = rand(Beta(PP[j]-alpha,PP_partial[j-1]-(j-1)*alpha))
    end
    return Psi
end
