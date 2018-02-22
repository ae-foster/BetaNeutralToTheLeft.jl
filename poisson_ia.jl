


function update_poisson_interarrival_param!(lambda::Vector{Float64},PP::Vector{Int},T::Vector{Int},n::Int,params::Vector{Float64})
    """
    - `p`: current geometric parameter for the interarrival time distribution
    - `K`: number of blocks in partition (=# of arrivals)
    - `n`: number of observations
    - `a`,`b`: parameters of prior Beta distribution
    """
    K = size(T,1)
    update_poisson_interarrival_param!(lambda,T[end],K,n,params)

end

function update_poisson_interarrival_param!(lambda::Vector{Float64},T_K::Int,K::Int,n::Int,params::Vector{Float64})
    """
    - `p`: current geometric parameter for the interarrival time distribution
    - `K`: number of blocks in partition (=# of arrivals)
    - `n`: number of observations
    - `a`,`b`: parameters of prior Beta distribution
    """
    # K = size(T,1)
    # sample pseudo arrival K+1 (given p, doesn't affect distribution of other arrival times)
    # T_Kp1 = n + 1 + rand(Truncated(Poisson(lambda[1]),n - T_K, Inf)) # this throws `InexactError()`; not sure why
    supp = 0:ceil(100*lambda[1])
    logp = logpdf(Poisson(lambda[1]),n - T_K .+ supp)
    T_Kp1 = n + 1 + wsample(supp,log_sum_exp_weights(logp))
    # sample conjugate lambda
    lambda[1] = rand(Gamma(T_Kp1-K-1+params[1],params[2]/(params[2]*K + 1)))
end
