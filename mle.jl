using Plots
using Optim
using StatsBase
include("dataset.jl")
plotly()

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>

function ntl_llikelihood(alpha::Vector{Float64}, ds::Vector{Int64},
        dcounts::Vector{Int64}, ts::Vector{Int64}, K::Int64, Tend::Int64)
    # For optim
    alpha = 1 - exp(alpha[1])
    if alpha >= 1
        return -Inf
    end
    lnum = -K*lgamma(1-alpha) + dcounts' * lgamma.(ds - alpha)

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
            ldenom += log(i-1-alpha*k)
        end
    end

    return lnum - ldenom
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
    glnum = K*digamma(1-alpha) - dcounts' * digamma.(ds - alpha)

    gldenom = 0
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
            gldenom -= k/(i-1-alpha*k)
        end
    end

    # negative
    storage[1] = -trans_correct * (glnum - gldenom)
end

function geom_llikelihood(g::Vector{Float64}, deltas::Vector{Int64})
    # for optim
    g = exp(g[1])/(1+exp(g[1]))
    return length(deltas)*log(g) + sum(deltas-1)*log(1-g)
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
    ldenom = sum( log.( 1:(Tend-1) + theta ) )

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

dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        println("$fname")
        degs, ts = parseSnapData("$dir$fname")
        Tend = sum(degs)
        dmap = countmap(degs)
        ds = collect(keys(dmap))
        dcounts = collect(values(dmap))
        deltas = ts[2:end] - ts[1:(end-1)]
        K = sum(dcounts)

        println("\nPYP")
        result = optimize(params -> -pyp_llikelihood(params, ds, dcounts, ts, K, Tend),
                          (storage, params) -> neg_grad_pyp_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                          [0., .5], LBFGS())
        #println(result)
        tau = exp(result.minimizer[1])/(1+exp(result.minimizer[1]))
        theta = result.minimizer[2]
        println("Tau ", tau, " Theta ", theta, )
        println("Optimized ll ", -result.minimum)
        println("Expected powerlaw ", 1+tau)

        println("NTL")
        result = optimize(a -> -ntl_llikelihood(a, ds, dcounts, ts, K, Tend),
                          (storage, params) -> neg_grad_ntl_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                          [0.], LBFGS())
        alpha = 1 - exp(result.minimizer[1])
        println("NTL alpha ", alpha)
        ll = result.minimum
        result = optimize(g -> -geom_llikelihood(g, deltas), [.5], LBFGS())
        println("g ", exp(result.minimizer[1])/(1+exp(result.minimizer[1])))
        g = length(degs)/sum(degs)
        println("g analytically ", g)
        ll += result.minimum
        println("Optimized ll ", -ll)
        println("Expected powerlaw ", 1+(1/g - alpha)/(1/g - 1))

        println("\n\n")
    end
end
