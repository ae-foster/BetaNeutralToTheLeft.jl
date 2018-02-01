###########################################################################
# Utilities for Gibbs updates
###########################################################################

"""
to do:
  - conjugate interarrival distn --> posterior predictive updates (need to carry around T_{K+1})
  - HMC or slice sampling updates for alpha parameter
  - CRP arrivals + theta updates
"""

# using StatsBase

function logp_partition(PP::Vector{Int},T::Vector{Int},Psi::Vector{Float64},
        alpha::Float64,ia_dist::DiscreteDistribution,is_partition::Bool)
    """
    - `PP`: vector of partition block sizes ordered by arrival time
    - `T`: vector of arrival times
    - `Psi`: vector of beta random variables (can be log(Psi))
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
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
    zero_shift = Int(minimum(ia_dist) == 0)

    PP_partial = cumsum(PP)
    # pop!(PP_partial)
    ia = T[2:end] .- T[1:(end-1)]

    K = size(Psi,1)
    idx = 1:(K-1)
    N = sum(PP)

    log_p = dot((PP[2:end] .- alpha .- 1),log_Psi[2:end]) + dot((PP_partial[1:(end-1)] .- idx.*alpha .- 1), log_Psi_c[2:end])
    # include arrival times
    log_p += sum(logpdf.(ia_dist, ia .- zero_shift))
    N - T[end] > 0 ? log_p += log(1 - cdf(ia_dist, N-T[end]-zero_shift)) : nothing
    log_p += -sum([lbeta(1 - alpha,T[j] - 1 - (j-1)*alpha) for j in 2:K])
    # include binomial coefficients if for a partition
    if is_partition
      log_p += sum([lbinom(PP_partial[j] - T[j],PP[j] - 1) for j in 2:K])
    end

    return log_p

end

function lbinom(n::Int,k::Int)
    """
    compues log of binomial coefficient {`n` choose `k`}
    """
    ret = lgamma(n+1) - lgamma(k+1) - lgamma(n - k + 1)
    return ret
end

function tally_ints(Z::Vector{Int},K::Int)
    """
    counts occurrences in `Z` of integers 1 to `K`
    - `Z`: vector of integers
    - `K`: maximum value to count occurences in `Z`
    """
    ret = zeros(Int,K)
    n = size(Z,1)
    idx_all = 1:n
    idx_j = trues(n)
    for j in 1:K
      for i in idx_all[idx_j]
        if Z[i]==j
          ret[j] += 1
          idx_j[i] = false
        end
      end
    end
    return ret
end


function seq2part(Z::Vector{Int})
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    """
    # transform Z into an ordered partition and get arrival times
    K = maximum(Z)
    PP = tally_ints(Z,K)

    # T = zeros(Int,K)
    # for j in 1:K
    #   T[j] = findfirst(Z.==j)
    # end
    return PP
end

function get_arrivals(Z::Vector{Int})
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    """

    K = maximum(Z)
    T = zeros(Int,K)
    for j in 1:K
      T[j] = findfirst(Z.==j)
    end
    return T
end

function logp_label_sequence(Z::Vector{Int},Psi::Vector{Float64},
        alpha::Float64,ia_dist::DiscreteDistribution)
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    - `Psi`: vector of beta random variables (can be log(Psi))
    - `alpha`: 'discount parameter' in size-biased reinforcement
    - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
    """

    PP = seq2part(Z)
    T = get_arrivals(Z)
    log_p = logp_partition(PP,T,Psi,alpha,ia_dist,false)
    return log_p

end

# condition on Psi version
# function update_label_sequence(Z::Vector{Int},Psi::Vector{Float64},
#           alpha::Float64,ia_dist::DiscreteDistribution)
#
#
# end

# marginalize Psi version
function update_z_pp_ss_t_cparams!(
  Z::Vector{Int},
  PP::Vector{Int},
  SS::Vector{Int},
  T::Vector{Int},
  theta::Array{Float64},
  X::Array{Float64},
  alpha::Float64,
  ia_dist::DiscreteDistribution,
  update_sufficient_stats!::Function,
  emission_logp::Function,
  sample_cluster_param::Function)
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition (from previous iteration, will be updated)
    - `PP`: arrival-ordered partition block sizes (from previous iteration, will be updated)
    - `SS`: arrival-ordered sufficient statistics per cluster (from previous iteration, will be updated)
    - `theta`: array of cluster parameters (for now, assumed to have one column per cluster,
        will also be updated as necessary (deletions and additions))
    - `T`: arrival times (from previous iteration, will be updated)
    - `X`: array of data; rows correspond to elements of `Z`
    - `alpha`: 'discount' parameter
    - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
    *** assumes fixed interarrival distribution
    - `update_sufficient_stats`: function that updates cluster sufficient statistics by removing an observation
    - `emission_logp`: function that returns marginalized log-emission probability
        input arguments should be:
        - `X_i`: the data corresponding to the current label being updated
        - `n_k`: the number of observations currently assigned to the cluster being calculated
        - `SS`: the sufficient statistics for the cluster being calculated
    - `sample_cluster_param`: function to generate parameters for new clusters
    """

    n = size(Z,1)
    zero_shift = Int(minimum(ia_dist) == 0)
    K_i = get_num_blocks(Z)
    # T = get_arrivals(Z)
    # PP = seq2part(Z)
    # Z_update = deepcopy(Z)
    # theta_update = deepcopy(theta)

    # compute sufficient statistics for each cluster
    # SS = compute_sufficient_stats(X,Z)

    K_n = size(T,1)

    for i in 2:n # Z[1] = 1 w.p. 1

      if PP[Z[i]]==1 # singleton (and therefore arrival time)
        deleteat!(PP,Z[i])
        deleteat!(T,K_i[i])
        K_i[i:end] += -1
        K_n += -1
        cluster_rm!(theta,Z[i],K_n)
        cluster_rm!(SS,Z[i],K_n)

      elseif T[K_i[i]]==i # arrival time but not singleton
        K = K_i[i]
        new_arrival = i + findfirst(Z[(i+1):end].==Z[i]) # reset arrival time for Z[i]'s cluster
        K_i[i:(new_arrival-1)] += -1 # decrement num_blocks for i <= j <= new_arrival
        # update arrival times
        insert_idx = findfirst(T .> new_arrival)
        insert_idx > 0 ? cycle_elements_left!(T,K,insert_idx-1) : cycle_elements_left!(T,K,size(T,1))
        insert_idx > 0 ? T[insert_idx-1] = new_arrival : T[end] = new_arrival
        # update partition
        PP[K] += -1
        insert_idx > 0 ? cycle_elements_left!(PP,K,insert_idx-1) : cycle_elements_left!(PP,K,size(PP,1))
        update_sufficient_stats!(SS,Z[i],-X[:,i])
        insert_idx > 0 ? cycle_elements_left!(SS,K,insert_idx-1) : cycle_elements_left!(SS,K,K_n)
        insert_idx > 0 ? cycle_elements_left!(theta,K,insert_idx-1) : cycle_elements_left!(theta,K,K_n)

      else # T, K_i, theta stay the same
        PP[Z[i]] += -1
        update_sufficient_stats!(SS,Z[i],-X[:,i])
      end

      Z[i] = 0

      ia = T[2:end] - T[1:(end-1)]
      # calculate log probabilities
      assert(K_n==K_i[end] && K_i[end]==maximum(K_i) && K_n==size(PP,1) && K_n==size(T,1))
      logp = zeros(Float64,K_n+1)
      for j in 1:K_i[i]
        # not proposing an arrival time
        logp[j] = log_CPPF(PP + [j==i for i in 1:K_n],T,alpha) + sum(logpdf.(ia_dist,ia .- zero_shift))
        T[end] == n ? nothing : logp[j] += log(1 - cdf(ia_dist,n - T[end] - zero_shift))
        logp[j] += emission_logp(X[:,i],PP[j],SS[j])
      end

      # proposing a new arrival time for a later cluster
      for j in (K_i[i]+1):K_n
        j<K_n ? T_prop = vcat(T[1:(j-1)],[i],T[(j+1):end]) : T_prop = vcat(T[1:(j-1)],[i])
        cycle_elements_right!(T_prop,K_i[i]+1,j)
        ia_prop = T_prop[2:end] - T_prop[1:(end-1)]
        logp[j] = cppf_counts(PP + [j==i for i in 1:K_n],alpha) + cppf_arrivals(T_prop,n,alpha)
        logp[j] += sum(logpdf.(ia_dist,ia_prop .- zero_shift))
        T_prop[end] == n ? nothing : logp[j] += log(1 - cdf(ia_dist,n - T_prop[end] - zero_shift))
        logp[j] += emission_logp(X[:,i],PP[j],SS[j])
      end

      # proposing a new cluster
      T_prop = vcat(T[1:K_i[i]],[i],T[(K_i[i]+1):end])
      ia_prop = T_prop[2:end] - T_prop[1:(end-1)]
      logp[K_n+1] = cppf_counts(PP,alpha) + cppf_arrivals(T_prop,n,alpha) + sum(logpdf.(ia_dist,ia_prop .- zero_shift))
      T_prop[end] == n ? nothing : logp[K_n+1] += log(1 - cdf(ia_dist,n - T_prop[end] - zero_shift))
      logp[K_n+1] = emission_logp(X[:,i],0,[])

      # sample cluster assignment
      p = log_sum_exp_weights(logp)
      c = wsample(1:size(logp,1),p)

      # update appropriate quantities
      if c <= K_i[i] # only need to update partition & sufficient stats
        Z[i] = c
        PP[c] += 1
        update_sufficient_stats!(SS,Z[i],X[:,i])
      elseif K_i[i] < c <= K_n # add to later cluster; update arrival times and labels
        # K_i[i:end] += 1
        K_i[i] += 1
        # update partition
        PP[c] += 1
        cycle_elements_right!(PP,K_i[i],c)
        # update arrivals
        T[c] = i
        cycle_elements_right!(T,K_i[i],c)
        # update labels
        Z[i] = K_i[i]
        Z[Z .== c] = K_i[i]
        Z[Z .> K_i[i]] += 1
        update_sufficient_stats!(SS,Z[i],X[:,i])
        cycle_elements_right!(SS,K_i[i],c)
        cycle_elements_right!(theta,K_i[i],c)
      else # create new cluster
        K_i[i:end] += 1
        Z[i] = K_i[i]
        insert!(PP,K_i[i],1)
        insert!(T,K_i[i],i)
        # draw new cluster parameter
        new_theta = sample_cluster_param()
        cluster_add!(theta,new_theta,K_i[i])
        cluster_add!(SS,X[:,i],K_i[i])

      end

    end
    return Z,PP,SS,T,theta

end

function cluster_rm!(x::Vector{Vector{Float64}},k::Int)
  """
  removes column k from vector of vectors x (e.g. corresponding cluster params)
  """
  deleteat!(x,k)
end

function cluster_rm!(x::Vector{Float64},k::Int)
  """
  removes k-th entry from x
  """
  deleteat!(x,k)
end

function cluster_add!(x::Vector{Vector{Float64}},x_new::Vector{Float64},k::Int)
  """
  inserts `x_new` into `x` at entry `k`
  """
  insert!(x,k,x_new)
end

function cluster_add!(x::Vector{Float64},x_new::Float64,k::Int)
  """
  inserts `x_new` into `x` at entry `k`
  """
  insert!(x,k,x_new)
end

function emission_diag_normal_conjugate(X::Vector{Float64},SS::Array{Float64})
    """
    - `X`: observation at which to calculated pdf
    - `SS`: current sufficient statistics [count,sum] of cluster to be computed
      (excluding current data point if necessary)
    """

end


function cycle_elements_left!(V::Vector,start_idx::Int,end_idx::Int)
    """
    - `V`: Vector whose elements will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    st = V[start_idx]
    for i in start_idx:(end_idx-1)
      V[i] = V[i+1]
    end
    V[end_idx] = st
    return V
end

function cycle_elements_left!(X::Array,start_idx::Int,end_idx::Int)
    """
    - `X`: Array whose columns will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    for i in 1:size(X,1)
      X[i,:] = cycle_elements_left!(X[i,:],start_idx,end_idx)
    end
    return X
end

function cycle_elements_right!(V::Vector,start_idx::Int,end_idx::Int)
    """
    - `V`: Vector whose elements will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    ed = V[end_idx]
    for i in end_idx:-1:(start_idx+1)
      V[i] = V[i-1]
    end
    V[start_idx] = ed
    return V

end

function cycle_elements_right!(X::Array,start_idx::Int,end_idx::Int)
    """
    - `X`: Array whose columns will be cycled
    - `start_idx`: start of cycle (will be moved to end)
    - `end_idx`: end of cycle
    """

    for i in 1:size(X,1)
      X[i,:] = cycle_elements_right!(X[i,:],start_idx,end_idx)
    end
    return X
end

function cppf_counts(PP::Vector{Int},alpha::Float64)
    """
    helper function for `log_CPPF` and `update_label_sequence`
    """
    lgam_1ma = lgamma(1 - alpha)
    ret = sum( lgamma.(PP[PP.>1] .- alpha) .- lgam_1ma )
    return ret
end

function cppf_arrivals(T::Vector{Int},n::Int,alpha::Float64)
    """
    helper function for `log_CPPF` and `update_label_sequence`
    """
    K = size(T,1)
    ret = -lgamma(n - K*alpha) + sum( lgamma.(T - (1:K).*alpha) ) - sum( lgamma(T[2:end] .- 1 - (0:(K-1)).*alpha) )
    return ret
end

function log_CPPF(PP::Vector{Int},T::Vector{Int},alpha::Float64)
    """
    - `PP`: arrival-ordered vector of partition block sizes
    - `T`: arrival times
    - `alpha`: 'discount' parameter
    """

    n = sum(PP)
    K = size(T,1)

    logp = cppf_arrivals(T,n,alpha) + cppf_counts(PP,alpha)
    return logp
end

function log_CPPF(Z::Vector{Int},alpha::Float64)
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    - `alpha`: 'discount' parameter
    """

    T = get_arrivals(Z)
    PP = seq2part(Z)
    logp = log_CPPF(PP,T,alpha)
    return logp
end

function get_num_blocks(Z::Vector{Int})
    """
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    """

    n = size(Z,1)
    K = zeros(Int64,n)
    K[1] = 1
    max_z = 1
    for i in 2:n
      if Z[i] > max_z
        K[i] = K[i-1] + 1
        max_z += 1
      end
    end
    return K
end

# function update_psi_parameters_sequence(Z::Vector{Int},alpha::Float64)
#     """
#     - `Z`: vector of labels corresponding to clusters/blocks in a partition
#     - `alpha`: 'discount' parameter
#     """
#
#     PP = seq2part(Z)
#     Psi = update_psi_parameters_partition(Psi,PP,alpha)
#     return Psi
#
# end

function update_psi_parameters_sequence!(Psi::Vector{Float64},Z::Vector{Int},alpha::Float64)
    """
    - `Psi`: vector of current values of Psi
    - `Z`: vector of labels corresponding to clusters/blocks in a partition
    - `alpha`: 'discount' parameter
    """

    PP = seq2part(Z)
    update_psi_parameters_partition!(Psi,PP,alpha)
    # Psi[:] = [pu[i] for i in 1:size(Psi,1)]
    return Psi

end

# function update_psi_parameters_partition(PP::Vector{Int},alpha::Float64)
#     """
#     - `PP`: arrival-ordered vector of partition block sizes
#     - `alpha`: 'discount' parameter
#     """
#     K = size(PP,1)
#     PP_partial = cumsum(PP)
#
#     Psi = zeros(Float64,K)
#     Psi[1] = 1
#     for j in 2:K
#       Psi[j] = rand(Beta(PP[j]-alpha,PP_partial[j-1]-(j-1)*alpha))
#     end
#     return Psi
# end

function update_psi_parameters_partition!(Psi::Vector{Float64},PP::Vector{Int},alpha::Float64)
    """
    - `Psi`: vector of current values of Psi
    - `PP`: arrival-ordered vector of partition block sizes
    - `alpha`: 'discount' parameter
    """
    K = size(PP,1)
    PP_partial = cumsum(PP)

    Psi[1] = 1
    for j in 2:K
      Psi[j] = rand(Beta(PP[j]-alpha,PP_partial[j-1]-(j-1)*alpha))
    end
    return Psi
end


function log_sum_exp_weights(logw::Vector{Float64})
  """
  -`logw`: log of weights to be combined for a discrete probability distribution
  """

  maxlogw = maximum(logw)
  shift_logw = logw - maxlogw
  p = exp.(shift_logw)./sum(exp.(shift_logw))
  return p
end

# function update_arrival_times(T::Vector{Int},PP::Vector{Int},alpha::Float64,ia_dist::DiscreteDistribution)
#     """
#     - `PP`: arrival-ordered vector of partition block sizes
#     - `T`: current arrival times (to be updated)
#     - `alpha`: 'discount' parameter
#     - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
#     *** assumes fixed interarrival distribution
#     """
#     T_update = deepcopy(T)
#     zero_shift = Int(minimum(ia_dist) == 0)
#     K = size(T,1)
#     PP_partial = cumsum(PP)
#     n = PP_partial[end]
#
#     for j in 2:(K-1)
#       delta2 = T_update[j+1] - T_update[j-1]
#       # determine support
#       supp = 1:min(delta2 - 1, PP_partial[j-1] - T_update[j-1] + 1)
#       # calculate pmf of conditional distribution
#       log_p = zeros(Float64,size(supp,1))
#       for s in supp
#         log_p[s] = logpdf(ia_dist,delta2 - (s-zero_shift)) + logpdf(ia_dist,s-zero_shift)
#         log_p[s] += lbinom(PP_partial[j] - T_update[j-1] - s, PP[j] - 1)
#         log_p[s] += lgamma(T_update[j-1] + s - j*alpha) - lgamma(T_update[j-1] + s - 1 - (j-1)*alpha)
#       end
#       # sample an update
#       p = log_sum_exp_weights(log_p)
#       T_update[j] = T_update[j-1] + wsample(supp,p)
#     end
#
#     # update final arrival time
#     supp = 1:min(n - T_update[K-1] - 1, PP_partial[K-1] - T_update[K-1] + 1)
#     log_p = zeros(Float64,size(supp,1))
#     for s in supp
#       log_p[s] = logpdf(ia_dist, s-zero_shift) + log(1 - cdf(ia_dist,n - T_update[K-1]-(s-zero_shift)))
#       log_p[s] += lbinom(n - T_update[K-1] - s, PP[K] - 1)
#       log_p[s] += lgamma(T_update[K-1] + s - K*alpha) - lgamma(T_update[K-1] + s - 1 - (K-1)*alpha)
#     end
#     p = log_sum_exp_weights(log_p)
#     T_update[K] = T_update[K-1] + wsample(supp,p)
#
#     return T_update
# end

function initialize_arrival_times(PP::Vector{Int},alpha::Float64,ia_dist::DiscreteDistribution)
    """
    - `PP`: arrival-ordered vector of partition block sizes
    - `alpha`: 'discount' parameter
    - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
    *** assumes fixed interarrival distribution
    """
    zero_shift = Int(minimum(ia_dist) == 0)
    K = size(PP,1)
    PP_partial = cumsum(PP)
    n = PP_partial[end]

    T = zeros(Int,K)
    T[1] = 1

    for j in 2:K
      # determine support of interarrival
      supp = 1:(PP_partial[j-1] - T[j-1] + 1)
      # calculate pmf of conditional distribution
      log_p = zeros(Float64,size(supp,1))
      for s in supp
        log_p[s] = logpdf(ia_dist,s-zero_shift)
        log_p[s] += lbinom(PP_partial[j] - T[j-1] - s, PP[j] - 1)
        log_p[s] += lgamma(T[j-1] + s - j*alpha) - lgamma(T[j-1] + s - 1 - (j-1)*alpha)
      end
      # sample an update
      p = log_sum_exp_weights(log_p)
      T[j] = T[j-1] + wsample(supp,p)
    end

    return T
end

function update_arrival_times!(T::Vector{Int},PP::Vector{Int},alpha::Float64,ia_dist::DiscreteDistribution)
    """
    - `PP`: arrival-ordered vector of partition block sizes
    - `T`: current arrival times (to be updated)
    - `alpha`: 'discount' parameter
    - `ia_dist`: distribution object corresponding to i.i.d. interarrivals
    *** assumes fixed interarrival distribution
    """
    # T_update = deepcopy(T)
    zero_shift = Int(minimum(ia_dist) == 0)
    K = size(T,1)
    PP_partial = cumsum(PP)
    n = PP_partial[end]

    for j in 2:(K-1)
      delta2 = T[j+1] - T[j-1]
      # determine support
      supp = 1:min(delta2 - 1, PP_partial[j-1] - T[j-1] + 1)
      # calculate pmf of conditional distribution
      log_p = zeros(Float64,size(supp,1))
      for s in supp
        log_p[s] = logpdf(ia_dist,delta2 - (s-zero_shift)) + logpdf(ia_dist,s-zero_shift)
        log_p[s] += lbinom(PP_partial[j] - T[j-1] - s, PP[j] - 1)
        log_p[s] += lgamma(T[j-1] + s - j*alpha) - lgamma(T[j-1] + s - 1 - (j-1)*alpha)
      end
      # sample an update
      p = log_sum_exp_weights(log_p)
      T[j] = T[j-1] + wsample(supp,p)
    end

    # update final arrival time
    if T[K-1]==(n-1)
      T[K] = n
    else
      supp = 1:min(n - T[K-1] - 1, PP_partial[K-1] - T[K-1] + 1)
      log_p = zeros(Float64,size(supp,1))
      for s in supp
        log_p[s] = logpdf(ia_dist, s-zero_shift) + log(1 - cdf(ia_dist,n - T[K-1]-(s-zero_shift)))
        log_p[s] += lbinom(n - T[K-1] - s, PP[K] - 1)
        log_p[s] += lgamma(T[K-1] + s - K*alpha) - lgamma(T[K-1] + s - 1 - (K-1)*alpha)
      end
      p = log_sum_exp_weights(log_p)
      T[K] = T[K-1] + wsample(supp,p)
    end

    return T
end

function update_geometric_interarrival_param_sequence(p::Float64,Z::Vector{Int},K::Int,a::Float64,b::Float64)
    """
    - `p`: current geometric parameter for the interarrival time distribution
    - `T`: arrival times
    - `n`: number of observations
    - `a`,`b`: parameters of prior Beta distribution
    """
    n = size(Z,1)
    # T = get_arrivals(Z)
    p_update = update_geometric_interarrival_param_partition(p,K,n,a,b)
    return p_update

end

function update_geometric_interarrival_param_partition(p::Float64,K::Int,n::Int,a::Float64,b::Float64)
    """
    - `p`: current geometric parameter for the interarrival time distribution
    - `K`: number of blocks in partition (=# of arrivals)
    - `n`: number of observations
    - `a`,`b`: parameters of prior Beta distribution
    """
    # K = size(T,1)
    # sample pseudo arrival K+1 (given p, doesn't affect distribution of other arrival times)
    T_Kp1 = rand(Geometric(p)) + n
    # sample conjugate p
    p_update = rand(Beta(K+a,T_Kp1-K-1+b))
    return p_update

end

function swap_elements!(x::Vector,i::Int,j::Int)
  """
  swap elements `i` and `j` of `x` in place
  """
  x[i],x[j] = x[j],x[i]
  return x
end

# needs to be tested (and made in-place)
function update_block_order!(perm::Vector{Int},PP::Vector{Int},T::Vector{Int},alpha::Float64)
    """
    - `PP`: current arrival-ordered vector of partition block sizes (to be updated)
    - `T`: arrival times
    - `alpha`: 'discount' parameter
    Update is through a sequence of proposed adjacent transpositions.

    There are likely better (more efficient) ways to do this.
    """

    K = size(PP,1)
    PP_update = PP[perm]
    PP_partial_update = cumsum(PP_update)

    for j in 1:(K-1)

      j==1 ? ppbar_jm1 = 0 : ppbar_jm1 = PP_partial_update[j-1]
      ppbar_jp1 = PP_partial_update[j+1]
      ppbar_j = PP_partial_update[j]
      pp_j = PP_update[j]
      pp_jp1 = PP_update[j+1]

      if ppbar_jm1 + pp_jp1 >= T[j+1] - 1
        logp_swap = lgamma(ppbar_jm1 + pp_jp1 - T[j] + 1) - lgamma(ppbar_jp1 - pp_j - T[j+1] + 2)
        logp_noswap = lgamma(ppbar_jm1 + pp_j - T[j] + 1) - lgamma(ppbar_j - T[j+1] + 2)
      else
        logp_swap = 0
        logp_noswap = 1
      end

      swap = wsample([true,false],[logp_swap,logp_noswap])
      if swap
        swap_elements!(PP_update,j,j+1)
        swap_elements!(perm,j,j+1)
        PP_partial_update[j] = (j == 1) ? PP_update[j] : PP_partial_update[j-1] + PP_update[j]
        PP_partial_update[j+1] = PP_partial_update[j] + PP_update[j+1]
      end

    end
    return perm
end
