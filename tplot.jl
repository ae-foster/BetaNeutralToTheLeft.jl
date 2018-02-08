using Plots
using StatsBase
include("dataset.jl")
plotly()

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>

# dir = "/data/flyrobin/foster/Documents/NTL.jl/"
dir = "data/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        degs, ts = parseSnapData("$dir$fname")
        fname = split(split(fname, ".txt")[1],"sx-")[end]
        println("\n$fname")

        # Plots.plot(ts, title=fname)
        # Plots.gui()
        #
        # Plots.histogram(ts[2:end] - ts[1:end-1], bins=100)
        # Plots.gui()

        # ef = ecdf(degs)
        # x = logspace(log(min(degs...)), log(max(degs...)), 100)
        # output = [(1-ef(t)) for t in x]
        # Plots.plot(x[output.>0], output[output.>0], xscale=:log10, yscale=:log10,
        #            title=fname)
        # Plots.gui()

        delta = ts[2:end] - ts[1:end-1]
        p_hat = length(delta)/sum(delta)
        ef = ecdf(delta)
        ks_stat = max([abs(1 - (1-p_hat)^k - ef(k)) for k=1:max(delta...)]...)
        efs = [ef(k) for k=1:max(delta...)]
        max_k = find(efs .< 0.98)[end]

        Plots.plot(1:max_k, [ef(k) for k=1:max_k], label = "True data", line=(3))
        Plots.plot!(1:max_k, [1 - (1-p_hat)^k for k=1:max_k], lw=3, label = "MLE", line=(3,:dash))
        Plots.plot!(title=fname, xlabel="Inter-arrival time", ylabel="Cumulative distribution function")
        Plots.gui()

        println("p_hat ", p_hat)
        println("KS D ", ks_stat)
        println("Critical ", 1.358/sqrt(length(delta)))

        # n = sum(degs)
        # xmin = 5
        # println("alpha hat ", 1 + n/(sum(log.(degs[degs.>xmin])) - n*log(xmin - 0.5)))
    end
end
