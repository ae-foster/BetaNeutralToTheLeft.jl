using Optim
using StatsBase
include("dataset.jl")
include("likelihoods.jl")

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>

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
