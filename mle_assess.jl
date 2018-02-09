using Plots
using JSON
using StatsBase
include("dataset.jl")
include("likelihoods.jl")
include("evaluation.jl")
include("crp.jl")
include("ntl_gibbs.jl")
# gr()
plotly()

## Assume serialized MLEs to be found in json file
percent = ARGS[1]
json_input = "mle_predict_$(percent)percent.json"
println("Reading predictions and truth from ", json_input)
f = open(json_input)
s = readstring(f)
results = JSON.parse(s)
println("Read successful")
plot_dir = "plots"
data_dir = "data/"

# for fname in readdir(data_dir)
fname = "sorted-mathoverflow.txt"
    if startswith(fname, "sorted-")
        fname_parsed = split(split(fname, ".txt")[1],"sorted-")[end]
        # if fname_parsed in ["CollegeMsg", "email", "stackoverflow"] continue end
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

        # nb_clusters_pyp_mean = [mean([sum(u .< t) for u in results[fname]["PYP"]["predicted arrivals"]]) for t in 1:length(T_new)]
        # nb_clusters_pyp = [[sum(u .< t) for u in results[fname]["PYP"]["predicted arrivals"]] for t in 1:length(T_new)]
        # nb_clusters_pyp_mean = [mean(x) for x in nb_clusters_pyp]
        # nb_clusters_pyp_std = [std(x) for x in nb_clusters_pyp]

        # n_test = results[fname]["n_test"] # right key even if it's seems wrong
        n_test = sum(results[fname]["true PP"]) - results[fname]["n_test"] # right key even if it's seems wrong
        nb_clusters_ntl = [[sum(u .< t) for u in results[fname]["NTL"]["predicted arrivals"]] for t in 1:n_test]
        nb_clusters_ntl_mean = [mean(x) for x in nb_clusters_ntl]
        # nb_clusters_ntl_std = [std(x) for x in nb_clusters_ntl]

        nb_clusters_true = [sum(T_new .< t) for t in 1:n_test]
        println(size(nb_clusters_ntl_mean))
        # println(size(nb_clusters_pyp_mean))
        println(size(nb_clusters_true))

        # plot(nb_clusters_pyp_mean, label="PYP")
        plot(nb_clusters_ntl, label="NTL")
        plot!(nb_clusters_true, color="black",lw=2, title="NTL_$fname_parsed")
        gui()

        # plot(1:min_size_ntl, T_new[1:min_size_ntl], label="Empirical", line=(2,:solid))
        # plot!(1:min_size_ntl, med_ntl, label="Predictive mean", linecolor="red", line=(2,:dashdot))
        # plot!(1:min_size_ntl, med_ntl+2*std_ntl, label="Predictive mean+2*std", linecolor="red", line=(1,:dot))
        # plot!(1:min_size_ntl, med_ntl-2*std_ntl, label="Predictive mean-2*std", linecolor="red", line=(1,:dot))
        # Plots.plot!(xlabel="Observations", ylabel="Arrival Time", guidefont = font(15), legendfont=font(12), legend = :topleft)
        # savefig("$plot_dir/predictive_arrival_times_NTL_$fname_parsed.pdf");
        # # gui()
        #
        # plot(1:min_size_pyp, T_new[1:min_size_pyp], label="Empirical", line=(2,:solid))
        # plot!(1:min_size_pyp, med_pyp, label="Predictive mean", linecolor="red", line=(2,:dashdot))
        # plot!(1:min_size_pyp, med_pyp+2*std_pyp, label="Predictive mean+2*std", linecolor="red", line=(1,:dot))
        # plot!(1:min_size_pyp, med_pyp-2*std_pyp, label="Predictive mean-2*std", linecolor="red", line=(1,:dot))
        # Plots.plot!(xlabel="Observations", ylabel="Arrival Time", guidefont = font(15), legendfont=font(12), legend = :topleft)
        # savefig("$plot_dir/predictive_arrival_times_PYP_$fname_parsed.pdf");
        # # gui()



        ### Adam's old code

        # pyp_predict = [Vector{Int64}(u) for u in results[fname]["PYP"]["predicted arrivals"]]
        # ntl_predict = [Vector{Int64}(u) for u in results[fname]["NTL"]["predicted arrivals"]]

        # plot(ntl_predict)
        # plot!(T_new,linecolor="black",lw=2, title="NTL_$fname_parsed")
        # gui()
        # plot(pyp_predict)
        # plot!(T_new,linecolor="black",lw=2, title="PYP_$fname_parsed")
        # gui()

        # One prediction for PP
        # n_test = results[fname]["n_test"]
        # PP_test = Vector{Int64}(results[fname]["true PP"])
        # n_train = sum(PP_test) - n_test
        # T_test = Vector{Int64}(results[fname]["true T"])
        # T = T_test[T_test .<= n_train]
        # ntl_PP = Vector{Int64}(results[fname]["NTL"]["PP"])
        # pyp_PP = Vector{Int64}(results[fname]["PYP"]["PP"])
        # ntl_T = Vector{Int64}(results[fname]["NTL"]["T"])
        # pyp_T = Vector{Int64}(results[fname]["PYP"]["T"])
        # println("TV(PYP, truth) ", total_variation_distance(pyp_PP, PP_test))
        # println("TV(NTL, truth) ", total_variation_distance(ntl_PP, PP_test))
        #
        # println("Assignments to new clusters ", sum(PP_test[length(T)+1:end]))
        # println("PYP Assignments to new clusters ", sum(pyp_PP[length(T)+1:end]))
        # println("NTL Assignments to new clusters ", sum(ntl_PP[length(T)+1:end]))
        #
        # println("#new clusters ", sum(T_test .> n_train))
        # println("PYP #new clusters ", sum(pyp_T .> n_train))
        # println("NTL #new clusters ", sum(ntl_T .> n_train))
        # dmap_true = countmap(PP_test)
        # d_true = collect(values(dmap_true))
        # dmap_pyp = countmap(pyp_PP)
        # d_pyp = collect(values(dmap_pyp))
        # dmap_ntl = countmap(ntl_PP)
        # d_ntl = collect(values(dmap_ntl))
        # println("degrees TV(PYP, truth) ", total_variation_distance(d_pyp, d_true))
        # println("degrees TV(NTL, truth) ", total_variation_distance(d_ntl, d_true))


        println("\n")
    end
# end
