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
syn_data_file = "./ess_output/gibbs_2018-03-05_22-01-23_syn_data.jld"
results_file = "./ess_output/gibbs_2018-03-05_22-01-23_ess_results.jld"
params_file = "./ess_output/gibbs_2018-03-05_22-01-23_params.json"


######################
# load results
######################
results = load(results_file)
syn_data = load(syn_data_file)
params = JSON.parsefile(params_file)


println(keys(results))

println(params["datasets"])
println(params["arrival_dists"])

n_ds = size(params["datasets"],1)
n_ad = size(params["arrival_dists"],1)


######################
# calculate statistics for table
######################

data_keys = ["alpha_mean_d";"slack_mean_d";"pred_ll_mean";"runtimes";"ESS_logp"]
data_keys_se = ["alpha_se_d";"slack_se_d";"pred_ll_se"]
# print_fmt_mn = ["\$%.1f\$";"\$%.2f\$";"\$%.3f\$";"\$%.2f\$";"\$%.2f\$";"\$%.2f\$"]
# # print_fmt_se = ["\$(%.1e)\$","\$%.2f\$","\$%.3f\$","\$%.2f\$","\$%.2f\$","\$%.2f\$"]
#
# sf = (s,t) -> @sprintf(s,t)
all_fmt = []
for d in 1:size(data_keys,1)

  mn = squeeze(mean(results[data_keys[d]],3),3)
  if d <= size(data_keys_se,1)
    se = squeeze(mean(results[data_keys_se[d]],3),3)
  else
    se = sqrt.(squeeze(var(results[data_keys[d]],3)./size(results[data_keys[d]],3),3))
  end

  mn_fmt = [@sprintf("\$%.3f", mn'[i]) for i in 1:prod(size(mn))]
  se_fmt = [@sprintf("%.3f\$",se'[i]) for i in 1:prod(size(se))]

  fmt = [join([mn_fmt[i],se_fmt[i]]," \\pm ") for i in 1:size(mn_fmt,1)]
  if d==1
    all_fmt = deepcopy(fmt)
  else
    all_fmt = [join([all_fmt[i],fmt[i]]," & ") for i in 1:size(fmt,1)]
  end

end

# data and models by hand: gen_alpha, gen_arrival_dist, n, K_n, model_arrival_dist
alpha_str = @sprintf("%.2f",params["ntl_alpha"])
theta_str = @sprintf("%.1f",params["ia_params"][1][1])
K_n_str_1 = @sprintf("%.0f",size(syn_data["K_data_all"][1],1))
beta_str = @sprintf("%.2f",params["ia_params"][2][1])
K_n_str_2 = @sprintf("%.0f",size(syn_data["K_data_all"][2],1))
gen_models = ["\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\tau,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\alpha,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\alpha,\\Geom(\\beta))\$ & ";
              "\$\\PYP($theta_str,$alpha_str)\$ & \$$K_n_str_1\$ & \$(\\alpha,\\Poisson_+(\\lambda))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\tau,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\alpha,\\PYP(\\theta,\\tau))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\alpha,\\Geom(\\beta))\$ & ";
              "\$\\Geom($beta_str)\$ & \$$K_n_str_2\$ & \$(\\alpha,\\Poisson_+(\\lambda))\$ & "]


out_txt = [join([gen_models[i],all_fmt[i]], "") for i in 1:size(all_fmt,1)]

open("./ess_output/tables/ess_table.txt", "w") do f
  for ln in 1:size(out_txt,1)
    write(f, join([out_txt[ln]," \\\\ \n \n"]))
  end
end
