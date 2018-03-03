using JLD
using JSON
using StatsBase

######################
# results filenames
######################
syn_data_file = "./ess_output/gibbs_2018-02-26_21-46-41_syn_data.jld"
results_file = "./ess_output/gibbs_2018-02-26_21-46-41_ess_results.jld"
params_file = "./ess_output/gibbs_2018-02-26_21-46-41_params.json"


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

data_keys = ["alpha_mean_d";"slack_mean_d";"runtimes";"ESS_sigma_logd"]
# print_fmt_mn = ["\$%.1f\$";"\$%.2f\$";"\$%.3f\$";"\$%.2f\$";"\$%.2f\$";"\$%.2f\$"]
# # print_fmt_se = ["\$(%.1e)\$","\$%.2f\$","\$%.3f\$","\$%.2f\$","\$%.2f\$","\$%.2f\$"]
#
# sf = (s,t) -> @sprintf(s,t)

for d in 1:size(data_keys,1)

  mn = squeeze(mean(results[data_keys[d]],3),3)
  se = sqrt.(squeeze(var(results[data_keys[d]],3)./size(results[data_keys[d]],3),3))

  mn_fmt = [@sprintf("\$%.2f", mn'[i]) for i in 1:prod(size(mn))]
  se_fmt = [@sprintf("%.2f\$",se'[i]) for i in 1:prod(size(se))]

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
K_n_str_1 = @sprintf("%.0f",size(syn_data["T_data_all"][1],1))
beta_str = @sprintf("%.2f",params["ia_params"][2][1])
K_n_str_2 = @sprintf("%.0f",size(syn_data["T_data_all"][2],1))
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
