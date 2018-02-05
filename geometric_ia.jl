

# function update_geometric_interarrival_param_sequence!(p::Vector{Float64},T::Vector{Int},n::Int,params::Vector{Float64})
#     """
#     - `p`: current geometric parameter for the interarrival time distribution
#     - `T`: arrival times
#     - `n`: number of observations
#     - `a`,`b`: parameters of prior Beta distribution
#     """
#     # n = size(Z,1)
#     # T = get_arrivals(Z)
#     update_geometric_interarrival_param_partition!(p,T,n,params)
#
# end

function update_geometric_interarrival_param!(p::Vector{Float64},T::Vector{Int},n::Int,params::Vector{Float64})
    """
    - `p`: current geometric parameter for the interarrival time distribution
    - `K`: number of blocks in partition (=# of arrivals)
    - `n`: number of observations
    - `a`,`b`: parameters of prior Beta distribution
    """
    K = size(T,1)
    update_geometric_interarrival_param!(p,K,n,params)

end

function update_geometric_interarrival_param!(p::Vector{Float64},K::Int,n::Int,params::Vector{Float64})
    """
    - `p`: current geometric parameter for the interarrival time distribution
    - `K`: number of blocks in partition (=# of arrivals)
    - `n`: number of observations
    - `a`,`b`: parameters of prior Beta distribution
    """
    # K = size(T,1)
    # sample pseudo arrival K+1 (given p, doesn't affect distribution of other arrival times)
    T_Kp1 = rand(Geometric(p[1])) + n
    # sample conjugate p
    p[1] = rand(Beta(K+params[1],T_Kp1-K-1+params[2]))

end
