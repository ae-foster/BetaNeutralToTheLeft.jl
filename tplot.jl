using Plots
using StatsBase
include("dataset.jl")
plotly()

## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>

data_fitted_parameters = Dict(
"superuser" => Dict(
"n" => 4209.0,
"n_tail" => 4209.0,
"xmin" => 86.0,
"eta" => 2.17946663001,
"D" => 0.00938960836573
),
"stackoverflow" => Dict(
"n" => 10327.0,
"n_tail" => 10327.0,
"xmin" => 1396.0,
"eta" => 2.46354520719,
"D" => 0.01176898838
),
"CollegeMsg" => Dict(
"n" => 50.0,
"n_tail" => 50.0,
"xmin" => 457.0,
"eta" => 4.09188189837,
"D" => 0.0614804797715
),
"email" => Dict(
"n" => 225.0,
"n_tail" => 225.0,
"xmin" => 917.0,
"eta" => 2.45504775514,
"D" => 0.0568919755977
),
"wiki-talk" => Dict(
"n" => 69148.0,
"n_tail" => 69148.0,
"xmin" => 14.0,
"eta" => 1.78809970004,
"D" => 0.0122889405739
),
"mathoverflow" => Dict(
"n" => 10405.0,
"n_tail" => 10405.0,
"xmin" => 8.0,
"eta" => 1.76753359359,
"D" => 0.0119082239774
),
"askubuntu" => Dict(
"n" => 21633.0,
"n_tail" => 21633.0,
"xmin" => 13.0,
"eta" => 2.14436457395,
"D" => 0.00612384834182
))

# dir = "/data/flyrobin/foster/Documents/NTL.jl/"
dir = "data/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        degs, ts = parseSnapData("$dir$fname")
        fname = split(split(fname, ".txt")[1],"-")[end]
        println("\n$fname")

        # Plots.plot(ts, title=fname)
        # Plots.gui()
        #
        # Plots.histogram(ts[2:end] - ts[1:end-1], bins=100)
        # Plots.gui()

        ef = ecdf(degs)
        x = logspace(log(min(degs...)), log(max(degs...)), 100)
        output = [(1-ef(t)) for t in x]
        p0 = Plots.plot(x[output.>0], output[output.>0], xscale=:log10, yscale=:log10, label = "True data", line=(3))
        xmin = Int(data_fitted_parameters[fname]["xmin"])
        a = -data_fitted_parameters[fname]["eta"] + 1
        b = 1-ef(log10(xmin)) - a * log10(xmin)
        lin_output = exp10.(a * log10.(x[output.>0]) + log10.(b))

        Plots.plot!(p0, x[output.>0], lin_output, xscale=:log10, yscale=:log10, label = "Fit", line=(3,:dash))
        Plots.plot!(p0, title=fname, xlabel="Node degree", ylabel="Counts")
        Plots.gui()

        delta = ts[2:end] - ts[1:end-1]
        p_hat = length(delta)/sum(delta)
        ef = ecdf(delta)
        ks_stat = max([abs(1 - (1-p_hat)^k - ef(k)) for k=1:max(delta...)]...)
        efs = [ef(k) for k=1:max(delta...)]
        max_k = find(efs .< 0.98)[end]

        p1 = Plots.plot(1:max_k, [ef(k) for k=1:max_k], label = "True data", line=(3))
        Plots.plot!(p1, 1:max_k, [1 - (1-p_hat)^k for k=1:max_k], label = "MLE", line=(3,:dash))
        Plots.plot!(p1, title=fname, xlabel="Inter-arrival time", ylabel="Cumulative distribution function")
        Plots.gui()

        p2 = Plots.plot(1:length(ts), ts, label="Arrivals", line=(3))
        Plots.plot!(p2, title=fname, xlabel="Observations", ylabel="Arrival Time")
        Plots.gui()

        println("p_hat ", p_hat)
        println("KS D ", ks_stat)
        println("Critical ", 1.358/sqrt(length(delta)))

        # n = sum(degs)
        # xmin = 5
        # println("alpha hat ", 1 + n/(sum(log.(degs[degs.>xmin])) - n*log(xmin - 0.5)))
    end
end
