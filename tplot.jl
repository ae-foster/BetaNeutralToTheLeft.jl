using Plots
using StatsBase
include("dataset.jl")

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>

dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        println(fname)
        degs, ts = parseSnapData("$dir$fname")

        Plots.plot(ts, title=fname)
        Plots.gui()
        #
        # Plots.histogram(ts[2:end] - ts[1:end-1], bins=100)
        # Plots.gui()

        ef = ecdf(degs)
        x = logspace(log(min(degs...)), log(max(degs...)), 100)
        output = [(1-ef(t)) for t in x]
        Plots.plot(x[output.>0], output[output.>0], xscale=:log10, yscale=:log10,
                   title=fname)
        Plots.gui()

        # n = sum(degs)
        # xmin = 5
        # println("alpha hat ", 1 + n/(sum(log.(degs[degs.>xmin])) - n*log(xmin - 0.5)))
    end
end
