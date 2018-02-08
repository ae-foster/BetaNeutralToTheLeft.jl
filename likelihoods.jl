# Helper function for sums excluding the ts's
function ldenom(f::Function, ts::Vector{Int64}, Tend::Int64, K::Int64)
    ldenom = 0
    t = ts[2]
    k = 1
    for i=2:Tend
        if i==t
            k+=1
            if k < K
                t=ts[k+1]
            else
                t=Inf
            end
        else
            ldenom += f(i, k)
        end
    end
    return ldenom
end

function ntl_llikelihood(params::Vector{Float64}, ds::Vector{Int64},
        dcounts::Vector{Int64}, ts::Vector{Int64}, K::Int64, Tend::Int64)
    """
    Computes the log-likelihood of the data conditional on arrival times (Tj).

    # Arguments
    - `params::Vector{Float64}`: a length 1 array containing log(1-alpha),
      a transform on the NTL alpha parameter
    - `ds::Vector{Int64}`: a vector of the unique observed cluster sizes
      (multiplicity should be stored in `dcounts`)
    - `dcounts::Vector{Int64}`: corresponding to `ds`, the number of times
      each cluster size occured in data. The orders of `dcounts` and `ds` must
      match
    - `ts::Vector{Int64}`: The vector of observed arrival times T1, ..., TK
       The first element will be 1.
    - `K::Int64`: the total number of clusters, K = length(ts)
    - `Tend::Int64`: the termination time of the data, Tend = ds'*dcounts
    """
    # For optim
    alpha = 1 - exp(params[1])
    if alpha >= 1
        return -Inf
    end
    if alpha < -1e10
        return -ldenom((i,k)->log(k), ts, Tend, K)
    end

    lnum = -K*lgamma(1-alpha) + dcounts' * lgamma.(ds - alpha)
    lden = ldenom(((i, k)->log(i-1-alpha*k)), ts, Tend, K)

    return lnum - lden
end

function neg_grad_ntl_llikelihood!(storage::Vector{Float64},
        params::Vector{Float64}, ds::Vector{Int64},
        dcounts::Vector{Int64}, ts::Vector{Int64}, K::Int64, Tend::Int64)
    # For optim
    alpha = 1 - exp(params[1])
    trans_correct = -exp(params[1])
    if alpha >= 1
        return Inf
    end
    if alpha < -1e10
        return 0
    end

    glnum = K*digamma(1-alpha) - dcounts' * digamma.(ds - alpha)
    gldenom = ldenom((i,k)->-k/(i-1-alpha*k), ts, Tend, K)

    # negative
    storage[1] = -trans_correct * (glnum - gldenom)
end

function geom_llikelihood(g::Vector{Float64}, deltas::Vector{Int64}, lag::Int64)
    # for optim
    g = exp(g[1])/(1+exp(g[1]))
    return length(deltas)*log(g) + (sum(deltas-1)+lag)*log(1-g)
end

function ntl_pyp_llikelihood(params::Vector{Float64}, ds::Vector{Int64},
        dcounts::Vector{Int64}, ts::Vector{Int64}, K::Int64, Tend::Int64)
    # For optim
    tau = exp(params[1])/(1+exp(params[1]))
    theta = params[2]
    alpha = params[3]
    if theta <= -tau
        return -Inf
    end
    lnum = lgamma(theta + 1) + sum( log.(theta .+ tau.*collect(1:(K-1))) ) + sum( lgamma.(ts[2:end] .- 1 .- tau.*collect(1:(K-1))) )
    ldenom =  lgamma(theta + ts[end]) + sum( lgamma.(ts[1:(end-1)] .- tau.*collect(1:(K-1))) )
    return lnum - ldenom + ntl_llikelihood([alpha],ds,dcounts,ts,K,Tend)
end

function pyp_llikelihood(params::Vector{Float64}, ds::Vector{Int64},
        dcounts::Vector{Int64}, ts::Vector{Int64}, K::Int64, Tend::Int64)
    # For optim
    tau = exp(params[1])/(1+exp(params[1]))
    theta = params[2]
    if theta <= -tau
        return -Inf
    end
    lnum = -K*lgamma(1-tau) + dcounts' * lgamma.(ds - tau) + sum(log.(theta+tau*collect(1:(K-1))))
    ldenom = sum( log.( (1:Tend-1) + theta ) )

    return lnum - ldenom
end

function neg_grad_pyp_llikelihood!(storage::Vector{Float64}, params::Vector{Float64},
        ds::Vector{Int64}, dcounts::Vector{Int64}, ts::Vector{Int64}, K::Int64,
        Tend::Int64)
    # For optim
    tau = exp(params[1])/(1+exp(params[1]))
    trans_correct = exp(params[1])/(1+exp(params[1]))^2
    theta = params[2]
    if theta <= -tau
        storage[1] = storage[2] = Inf
        return
    end

    storage[1] = -trans_correct * ( K*digamma(1-tau) - dcounts' * digamma.(ds - tau) + sum( k/(theta+tau*k) for k=1:(K-1) ) )
    storage[2] = -( sum( 1./(theta .+ tau*collect(1:(K-1))) ) - sum( 1./(theta .+ (1:(Tend-1))) ) )
end
