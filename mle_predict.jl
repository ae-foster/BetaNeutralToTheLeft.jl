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
json_input = "mle_results_$(percent)percent.json"
json_output = "mle_predict_$(percent)percent.json"
results = Dict()
println("Reading MLEs from ", json_input)
f = open(json_input)
s = readstring(f)
mles = JSON.parse(s)
println("Read successful")
split_prop = parse(Float64, percent)/100
dir = "/data/flyrobin/foster/Documents/NTL.jl/"
for fname in readdir(dir)
    if startswith(fname, "sorted-")
        if startswith(fname, "sorted-stackoverflow.txt")
            continue
        end
        println("$fname")
        results[fname] = Dict()
        PP, T, PP_test, T_test, n_train, n_test = trainTestSplitSnapData("$dir$fname", split_prop)
        # PP_test = 2*ones(Int64, 100)
        # T_test = collect(1:100)
        Tend = sum(PP_test)
        dmap = countmap(PP_test)
        ds = collect(keys(dmap))
        dcounts = collect(values(dmap))
        deltas = T_test[2:end] - T_test[1:(end-1)]
        lag = Tend-T_test[end]
        K = sum(dcounts)

        #######################################################################
        # Prediction
        #######################################################################

        # Load PYP
        println("PYP")
        results[fname]["PYP"] = Dict()
        tau = mles[fname]["PYP"]["tau"]
        theta = mles[fname]["PYP"]["theta"]
        train_ll = mles[fname]["PYP"]["ll"]
        println("Tau ", tau, " theta ", theta, " train ll ", train_ll)

        # Predictive log-likelihood
        predictive_ll = pyp_llikelihood([log(tau/(1-tau)), theta], ds, dcounts, T_test, K, Tend) - train_ll
        println("pll ", predictive_ll)
        ben_pll = predictive_logprob(PP,T,PP_test,T_test,CRP(theta,tau),tau)
        println("Ben pll ", ben_pll)
        results[fname]["PYP"]["pll"] = predictive_ll

        # Load NTL
        println("NTL")
        results[fname]["NTL"] = Dict()
        g = mles[fname]["NTL"]["g"]
        alpha = mles[fname]["NTL"]["alpha"]
        train_ll = mles[fname]["NTL"]["ll"]
        println("g ", g, " alpha ", alpha, " train ll ", train_ll)
        #
        # Predictive log-likelihood
        predictive_ll = ntl_llikelihood([log(1-alpha)], ds, dcounts, T_test, K, Tend) + geom_llikelihood([log(g/(1-g))], deltas, lag) - train_ll
        println("Predictive ll ", predictive_ll)
        ben_pll = predictive_logprob(PP,T,PP_test,T_test,(a,b)->Geometric(g),alpha)
        println("Ben pll ", ben_pll)
        results[fname]["NTL"]["pll"] = predictive_ll
        println("\n")

        ## Predict T
        ntl_predict = [sample_predicted_arrival_times((a,b)->Geometric(g),T[end],length(PP),n_train,n_test)-n_train for i=1:100]
        pyp_predict = [sample_predicted_arrival_times(CRP(theta, tau),T[end],length(PP),n_train,n_test)-n_train for i=1:100]
        # plot(T_new,linecolor="black",lw=2)
        # plot!(ntl_predict)
        # gui()
        # plot(T_new,linecolor="black",lw=2)
        # plot!(pyp_predict)
        # gui()
        results[fname]["true arrivals"] = T_new
        results[fname]["PYP"]["predicted arrivals"] = pyp_predict
        results[fname]["NTL"]["predicted arrivals"] = ntl_predict


        # Predict future cluster sizes and arrival times
        println("Data points to be assigned ", n_test)
        ntl_PP,ntl_T = sample_predicted_partition(PP,T[end],(a,b)->Geometric(g),alpha,n_test)
        pyp_PP,pyp_T = sample_predicted_partition(PP,T[end],CRP(theta,tau),tau,n_test)
        #println(ntl_predict)
        println("TV(NTL, truth) ", total_variation_distance(ntl_PP, PP_test))
        println("TV(PYP, truth) ", total_variation_distance(pyp_PP, PP_test))
        # println("Assignments to new clusters ", sum(PP_test[length(T):end]))
        # println("NTL Assignments to new clusters ", sum(ntl_predict[length(T):end]))
        # println("PYP Assignments to new clusters ", sum(pyp_predict[length(T):end]))
        # println("#new clusters ", length(PP_test) - length(PP))
        # println("NTL #new clusters ", length(ntl_predict) - length(PP))
        # println("pyp #new clusters ", length(pyp_predict) - length(PP))
        results[fname]["n_test"] = n_test
        results[fname]["true PP"] = PP_test
        results[fname]["true T"] = T_test
        results[fname]["NTL"]["PP"] = ntl_PP
        results[fname]["PYP"]["PP"] = pyp_PP
        results[fname]["NTL"]["T"] = ntl_T
        results[fname]["PYP"]["T"] = pyp_T

    end
end
open(json_output, "w") do f
    println("Writing to file ", json_output)
    write(f, JSON.json(results))
    println("Success")
end
