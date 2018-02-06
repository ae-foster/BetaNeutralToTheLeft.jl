using Plots
using StatsBase
include("dataset.jl")
plotly()

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>

dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        println("\n$fname")
        degs, ts = parseSnapData("$dir$fname")

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

        Plots.plot([ef(k) for k=1:max(delta...)], title=fname)
        Plots.plot!([1 - (1-p_hat)^k for k=1:max(delta...)])
        Plots.gui()

        println("p_hat ", p_hat)
        println("KS D ", ks_stat)
        println("Critical ", 1.358/sqrt(length(delta)))

        # n = sum(degs)
        # xmin = 5
        # println("alpha hat ", 1 + n/(sum(log.(degs[degs.>xmin])) - n*log(xmin - 0.5)))
    end
end
