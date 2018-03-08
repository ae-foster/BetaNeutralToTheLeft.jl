using JLD
using JSON
using StatsBase

include("dataset.jl")
include("crp.jl")
include("geometric_ia.jl")
include("poisson_ia.jl")
include("ntl_gibbs.jl")
include("slice.jl")
include("evaluation.jl")
include("gibbs_util.jl")
include("ess_experiments_util.jl")

######################
# results filenames
######################
syn_data_file = "./ess_output/gibbs_2018-03-06_18-20-48_syn_data_n_scale.jld"
results_file = "./ess_output/gibbs_2018-03-06_18-20-48_ess_results_n_scale.jld"
params_file = "./ess_output/gibbs_2018-03-06_18-20-48_params.json"


######################
# load results
######################
results = load(results_file)
syn_data = load(syn_data_file)
params = JSON.parsefile(params_file)


println(keys(results))

println(params["dataset"])
println(params["arrival_dist"])

n_ds = size(params["dataset"],1)
n_ad = size(params["arrival_dist"],1)
n_sub = convert(Vector{Int64},params["n_sub"])./2 # convert to number of edges


######################
# calculate statistics for table
######################

data_keys = ["alpha_mean_d";"ia_param_mean_d";"slack_mean_d";"runtimes";"ESS_logp";"ESS_alpha";"ESS_sigma_logd";"ESS_slack_logd"]
data_keys_se = ["alpha_se_d";"ia_param_se_d";"slack_se_d"]
# print_fmt_mn = ["\$%.1f\$";"\$%.2f\$";"\$%.3f\$";"\$%.2f\$";"\$%.2f\$";"\$%.2f\$"]
# # print_fmt_se = ["\$(%.1e)\$","\$%.2f\$","\$%.3f\$","\$%.2f\$","\$%.2f\$","\$%.2f\$"]
#
# sf = (s,t) -> @sprintf(s,t)
n_fmt = [@sprintf("\$%.0f\$",n_sub[n]) for n in 1:length(n_sub)]
all_fmt = join(n_fmt, " & ")

for d in 1:size(data_keys,1)

  startswith(data_keys[d],"ia_param") ? mn = squeeze(mean(results[data_keys[d]],3),3) : mn = mean(results[data_keys[d]],2)
  if startswith(data_keys[d],"ia_param")
    se = squeeze(mean(results[data_keys_se[d]],3),3)
  elseif d <= size(data_keys_se,1)
    se = mean(results[data_keys_se[d]],2)
  else
    se = sqrt.(var(results[data_keys[d]],2)./size(results[data_keys[d]],2))
  end

  mn_fmt = [@sprintf("\$%.3f", mn[i]) for i in 1:length(mn)]
  se_fmt = [@sprintf("%.3f\$",se[i]) for i in 1:length(se)]

  fmt = [join([mn_fmt[i],se_fmt[i]]," \\pm ") for i in 1:length(mn_fmt)]

  all_fmt = [all_fmt; join(fmt, " & ")]

end

# data and models by hand: gen_alpha, gen_arrival_dist, n, K_n, model_arrival_dist
alpha_str = @sprintf("%.2f",params["ntl_alpha"])
beta_str = @sprintf("%.2f",params["ia_params"][1][1])
K_n_str_1 = @sprintf("%.0f",syn_data["K_data_all"][1])
K_n_str_2 = @sprintf("%.0f",syn_data["K_data_all"][2])
K_n_str_3 = @sprintf("%.0f",syn_data["K_data_all"][3])

stats = ["\$|\\hat{\\alpha} - \\alpha^*|\$";
          "\$|\\hat{\\beta} - \\beta^*|\$"];
          "\$|\\hat{\\mathbf{S}} - \\mathbf{S}^*|\$";
          "Runtime (s)";
          "ESS(\$\\log(p)\$)";
          "ESS(\$\\alpha\$)";
          


gen_models = ["\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\tau,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\alpha,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\alpha,\\Geom(\\beta))\$ & ";
              "\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\alpha,\\Poisson_+(\\lambda))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\tau,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\alpha,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\alpha,\\Geom(\\beta))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\alpha,\\Poisson_+(\\lambda))\$ & "]


out_txt = [join([gen_models[i],all_fmt[i]], "") for i in 1:size(all_fmt,1)]

open("./ess_output/tables/ess_table_n_scale.txt", "w") do f
  for ln in 1:size(out_txt,1)
    write(f, join([out_txt[ln]," \\\\ \n \n"]))
  end
end
