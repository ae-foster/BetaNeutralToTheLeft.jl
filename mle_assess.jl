using Plots
using JSON
using StatsBase
include("dataset.jl")
include("likelihoods.jl")
include("evaluation.jl")
include("crp.jl")
include("ntl_gibbs.jl")
plotly()

## Assume serialized MLEs to be found in json file
percent = ARGS[1]
json_input = "mle_predict_$(percent)percent.json"
println("Reading predictions and truth from ", json_input)
f = open(json_input)
s = readstring(f)
results = JSON.parse(s)
println("Read successful")
dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        if startswith(fname, "sorted-stackoverflow.txt")
            continue
        end
        println(fname)

        #######################################################################
        # Assessment
        #######################################################################

        pyp_predictive_ll = results[fname]["PYP"]["pll"]
        println("PYP pll ", pyp_predictive_ll)
        ntl_predictive_ll = results[fname]["NTL"]["pll"]
        println("NTL pll ", ntl_predictive_ll)

        # Many predictions for Tj
        T_new = results[fname]["true arrivals"]
        pyp_predict = [Vector{Int64}(u) for u in results[fname]["PYP"]["predicted arrivals"]]
        ntl_predict = [Vector{Int64}(u) for u in results[fname]["NTL"]["predicted arrivals"]]

        plot(ntl_predict)
        plot!(T_new,linecolor="black",lw=2)
        gui()
        plot(pyp_predict)
        plot!(T_new,linecolor="black",lw=2)
        gui()

        # One prediction for PP
        n_test = results[fname]["n_test"]
        PP_test = Vector{Int64}(results[fname]["true PP"])
        n_train = sum(PP_test) - n_test
        T_test = Vector{Int64}(results[fname]["true T"])
        T = T_test[T_test .<= n_train]
        ntl_PP = Vector{Int64}(results[fname]["NTL"]["PP"])
        pyp_PP = Vector{Int64}(results[fname]["PYP"]["PP"])
        ntl_T = Vector{Int64}(results[fname]["NTL"]["T"])
        pyp_T = Vector{Int64}(results[fname]["PYP"]["T"])
        println("TV(PYP, truth) ", total_variation_distance(pyp_PP, PP_test))
        println("TV(NTL, truth) ", total_variation_distance(ntl_PP, PP_test))

        println("Assignments to new clusters ", sum(PP_test[length(T)+1:end]))
        println("PYP Assignments to new clusters ", sum(pyp_PP[length(T)+1:end]))
        println("NTL Assignments to new clusters ", sum(ntl_PP[length(T)+1:end]))

        println("#new clusters ", sum(T_test .> n_train))
        println("PYP #new clusters ", sum(pyp_T .> n_train))
        println("NTL #new clusters ", sum(ntl_T .> n_train))
        dmap_true = countmap(PP_test)
        d_true = collect(values(dmap_true))
        dmap_pyp = countmap(pyp_PP)
        d_pyp = collect(values(dmap_pyp))
        dmap_ntl = countmap(ntl_PP)
        d_ntl = collect(values(dmap_ntl))
        println("degrees TV(PYP, truth) ", total_variation_distance(d_pyp, d_true))
        println("degrees TV(NTL, truth) ", total_variation_distance(d_ntl, d_true))


        println("\n")
    end
end
