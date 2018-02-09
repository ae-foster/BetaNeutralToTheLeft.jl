using StatsBase
using JSON


## Download the unweighted datasets here https://snap.stanford.edu/data/#temporal
## gunzip <name>
## sort -k3 -n <name> > sorted-<name>
percent = ARGS[1]
json_input = "mle_results_$(percent)percent.json"
f = open(json_input)
results = JSON.parse(readstring(f))
dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        print(fname)
        print("  ")

        # PYP
        #println(result)
        tau = results[fname]["PYP"]["tau"]
        theta = results[fname]["PYP"]["theta"]
        ll = results[fname]["PYP"]["ll"]
        print(signif(tau,4), "   ", signif(theta,4), "   ", signif(ll,4), "   ", signif(tau+1,4), "  ")


        alpha = results[fname]["NTL"]["alpha"]
        g = results[fname]["NTL"]["g"]
        ll=results[fname]["NTL"]["ll"]
        print(signif(alpha,4), "  ", signif(g,4), "  ", signif(ll,4), "  ", signif(1+(1/g - alpha)/(1/g - 1),4), "  ")

        tau = results[fname]["Frankenstein"]["tau"]
        theta = results[fname]["Frankenstein"]["theta"]
        ll=results[fname]["Frankenstein"]["ll"]
        print(signif(tau,4), "  ", signif(theta,4), "  ", signif(-ll,4), "  ", signif(1+tau,4))
        println()

    end
end
