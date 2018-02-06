using Plots
using Optim
using StatsBase
include("dataset.jl")
include("likelihoods.jl")
include("evaluation.jl")
plotly()

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>

dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        println("$fname")
        degs, ts, degs_test, ts_test, n_test = trainTestSplitSnapData("$dir$fname")
        Tend = sum(degs)
        dmap = countmap(degs)
        ds = collect(keys(dmap))
        dcounts = collect(values(dmap))
        deltas = ts[2:end] - ts[1:(end-1)]
        K = sum(dcounts)

        #######################################################################
        # Fit MLEs for two cases
        #######################################################################

        println("\nPYP")
        result = optimize(params -> -pyp_llikelihood(params, ds, dcounts, ts, K, Tend),
                          (storage, params) -> neg_grad_pyp_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                          [0., .5], LBFGS())
        #println(result)
        tau = exp(result.minimizer[1])/(1+exp(result.minimizer[1]))
        theta = result.minimizer[2]
        println("Tau ", tau, " Theta ", theta, )

        println("NTL")
        result = optimize(a -> -ntl_llikelihood(a, ds, dcounts, ts, K, Tend),
                          (storage, params) -> neg_grad_ntl_llikelihood!(storage, params, ds, dcounts, ts, K, Tend),
                          [0.], LBFGS())
        alpha = 1 - exp(result.minimizer[1])
        println("NTL alpha ", alpha)
        g = length(degs)/sum(degs)
        println("g analytically ", g)

        #######################################################################
        # Prediction
        #######################################################################

        ntl_predict = 1/g:1/g:max(ts_test...)
        plot(ts_test)
        plot!(ntl_predict)
        gui()


        println("\n\n")
    end
end
