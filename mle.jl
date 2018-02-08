using Optim
using StatsBase
using JSON
include("dataset.jl")
include("likelihoods.jl")

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>
percent = ARGS[1]
json_output = "mle_results_$(percent)percent.json"
split_prop = parse(Float64, percent)/100
results = Dict()
dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        println(fname)
        results[fname] = Dict()
        degs, ts, _, _, _, _ = trainTestSplitSnapData("$dir$fname", split_prop)
        Tend = sum(degs)
        dmap = countmap(degs)
        ds = collect(keys(dmap))
        dcounts = collect(values(dmap))
        deltas = ts[2:end] - ts[1:(end-1)]
        lag=Tend-ts[end]
        K = sum(dcounts)

        println("\nPYP")
        results[fname]["PYP"] = Dict()
        result = optimize(params -> -pyp_llikelihood(params, ds, dcounts, ts, K, Tend),
                          (storage, params) -> neg_grad_pyp_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                          [0., .5], LBFGS())
        #println(result)
        tau = exp(result.minimizer[1])/(1+exp(result.minimizer[1]))
        theta = result.minimizer[2]
        println("Tau ", tau, " Theta ", theta, )
        results[fname]["PYP"]["tau"] = tau
        results[fname]["PYP"]["theta"] = theta
        println("Optimized ll ", -result.minimum)
        results[fname]["PYP"]["ll"] = -result.minimum
        println("Expected powerlaw ", 1+tau)

        println("NTL")
        results[fname]["NTL"] = Dict()
        result = optimize(a -> -ntl_llikelihood(a, ds, dcounts, ts, K, Tend),
                          (storage, params) -> neg_grad_ntl_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                          [0.], LBFGS())
        alpha = 1 - exp(result.minimizer[1])
        println("NTL alpha ", alpha)
        results[fname]["NTL"]["alpha"] = alpha
        ll = result.minimum
        result = optimize(g -> -geom_llikelihood(g, deltas, lag), [.5], LBFGS())
        g = exp(result.minimizer[1])/(1+exp(result.minimizer[1]))
        println("g ", g)
        results[fname]["NTL"]["g"] = g
        ll += result.minimum
        println("Optimized ll ", -ll)
        results[fname]["NTL"]["ll"] = -ll
        println("Expected powerlaw ", 1+(1/g - alpha)/(1/g - 1))

        println("\n")
    end
end
open(json_output, "w") do f
    println("Writing to file ", json_output)
    write(f, JSON.json(results))
    println("Success")
end
