using JLD
using JSON
using Plots
using DataStructures

gr()
# plotly()

dirname = "sampler_output/"

for fname_full in readdir(dirname)
    if endswith(fname_full, ".json")

fname = split(fname_full, "params")[1]
println("Computing infered parameters for $fname")
samples = load(dirname * fname * "samples.jld")

open(dirname * fname * "params.json", "r") do f
    global params
    dicttxt = readstring(f)  # file information to string
    params=JSON.parse(dicttxt)  # parse and transform data
end

if !(((params["arrivals"] == "crp") && (params["dataset_name"] == "synthetic crp")) || ((params["arrivals"] == "geometric") && (params["dataset_name"] == "synthetic geometric")))
    println("skip since model and dataset do not match")
    continue
end

    mean_alpha = mean(samples["alpha_gibbs"])
    std_alpha = std(samples["alpha_gibbs"])

    infered_params = OrderedDict(
    "alpha" => OrderedDict(
        "true" => params["ntl_alpha"],
        "mean" => mean_alpha,
        "std" => std_alpha
    ))

    if (params["arrivals"] == "crp") && (params["dataset_name"] == "synthetic crp")
        println("crp")

        mean_crp_theta = mean(samples["ia_params_gibbs"][1,:])
        std_crp_theta = std(samples["ia_params_gibbs"][1,:])

        infered_params["crp_theta"] = OrderedDict(
            "true" => params["crp_theta"],
            "mean" => mean_crp_theta,
            "std" => std_crp_theta
        )

        mean_crp_alpha = mean(samples["ia_params_gibbs"][2,:])
        std_crp_alpha = std(samples["ia_params_gibbs"][2,:])

        infered_params["crp_alpha"] = OrderedDict(
        "true" => params["crp_alpha"],
        "mean" => mean_crp_alpha,
        "std" => std_crp_alpha
        )

    elseif (params["arrivals"] == "geometric") && (params["dataset_name"] == "synthetic geometric")
        println("geometric")

        scatter(samples["alpha_gibbs"], samples["ia_params_gibbs"]', marker=(:circle, 3,0.6,:darkblue), label="Samples");
        scatter!([params["ntl_alpha"]], [params["geom_p"]], marker=(:diamond, 6,1.,:red), label="Thruth");
        plot!(title="Geometric", xlabel=" \\alpha ", ylabel=" \\beta", guidefont = font(15), legendfont = font(12), legend = :topleft);
        savefig(dirname * fname * "scatterplot.pdf");

        mean_geom_p = mean(samples["ia_params_gibbs"])
        std_geom_p = std(samples["ia_params_gibbs"])

        infered_params["geom_p"] = OrderedDict(
            "true" => params["geom_p"],
            "mean" => mean_geom_p,
            "std" => std_geom_p
        )
    end

    open(dirname * fname * "infered_params.json", "w") do f
        write(f, JSON.json(infered_params))
    end
end # if json

end # loop
