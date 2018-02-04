# Gibbs samplers for Beta NTL models of partitions, graphs, mixtures
using Distributions
using StatsBase

include("dataset.jl")
include("ntl_gibbs.jl")
include("slice.jl")


###########################################################################
# Gibbs sampler settings
###########################################################################
n_iter = 10000  # total number of Gibbs iterations to run
n_burn = 5000   # burn-in
n_thin = 10     # collect every `n_thin` samples

# set which components to update
gibbs_psi = true            # NTL Ψ paramters
gibbs_alpha = true          # NTL alpha parameter
gibbs_arrival_times = true  # arrival times
gibbs_ia_params = true     # arrival time distribution parameters
gibbs_perm_order = true   # order of blocks in partition/vertices in graph

## SET SEED

############################################################################
# Data
############################################################################
# need to add support for certain real data sets

dataset_name = "synthetic crp"

if dataset_name=="synthetic crp" # Synthetic data w/ CRP interarrivals
  include("crp.jl")
  N = 200
  ntl_alpha = 0.5
  crp_theta = 10.0
  crp_alpha = 0.1
  # create intearrival distribution object and synthetic data
  interarrival_dist = CRP(crp_theta,crp_alpha)
  Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist)

  gibbs_ia_params ? nothing : arrival_params_fixed = [crp_theta; crp_alpha]

  # function check
  assert(all(PP_data .== seq2part(Z_data)))
  assert(all(T_data .== get_arrivals(Z_data)))

elseif dataset_name=="synthetic geometric" # Synthetic data w/ geometric interarrivals
  N = 200
  ntl_alpha = 0.5
  p = 0.25
  # create intearrival distribution object and synthetic data
  interarrival_dist = Geometric
  Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist(p))

  gibbs_ia_params ? nothing : arrival_params_fixed = [p]

  # function check
  assert(all(PP_data .== seq2part(Z_data)))
  assert(all(T_data .== get_arrivals(Z_data)))

end

# sort partition if necessary
if gibbs_perm_order
  PP_sort = sortrows(hcat(PP_data,collect(1:size(PP_data,1))),rev=true)
  perm_data = PP_sort[:,2]
  PP = PP_sort[:,1]
else
  PP = deepcopy(PP_data)
end

gibbs_alpha ? nothing : alpha_fixed = ntl_alpha

###########################################################################
# NTL alpha settings
###########################################################################
ntl_gamma_a = 1.
ntl_gamma_b = 1.

# prior is specified as a distribution on (0,Inf);
#   transformation to alpha ∈ (-Inf,1) will be performed during sampling as appropriate
ntl_alpha_prior_dist = Gamma(ntl_gamma_a,ntl_gamma_b)
ntl_alpha_log_prior = x -> logpdf(ntl_alpha_prior_dist,x)

w_alpha = 1.0 # slice sampling "window" parameter


###########################################################################
# Arrival time distribution settings
###########################################################################
arrivals = "geometric"

if arrivals=="crp"
  include("crp.jl")
  # CRP arrival distribution
  ia_dist = v -> CRP(v[1],v[2])

  # prior on CRP parameters
  n_ia_params = 2
  theta_gamma_a = 1.
  theta_gamma_b = 4.
  alpha_beta_a = 1.
  alpha_beta_b = 1.
  ia_prior_params = [theta_gamma_a; # prior on theta
                     theta_gamma_b;
                     alpha_beta_a; # prior on crp_alpha
                     alpha_beta_b]

  # slice sampling functions/parameters
  f_lp_t = (x,a) -> logpdf(Gamma(theta_gamma_a,theta_gamma_b),x+a)
  f_lp_a = x -> logpdf(Beta(alpha_beta_a,alpha_beta_b),x)
  w_t = 1.0 # slice sampling w parameter for crp_theta
  w_a = 1.0 # slice sampling w parameter for crp_alpha

  # CRP-specific sampling functions
  initialize_arrival_params = v -> initialize_crp_params(Gamma(v[1],v[2]),Beta(v[3],v[4]))
  update_arrival_params! = (ap,tt,n,pp) -> update_crp_interarrival_params!(ap,tt,n,f_lp_t,f_lp_a,w_t,w_a)


elseif arrivals=="geometric"
  include("geometric_ia.jl")
  # Geometric interarrival distribution
  ia_dist = p -> Geometric(p[1])
  a_beta = 1.
  b_beta = 1.
  ia_prior_params = [a_beta; b_beta]
  n_ia_params = 1
  # set update functions
  initialize_arrival_params = v -> [rand(Beta(v[1],v[2]))]
  update_arrival_params! = update_geometric_interarrival_param!

end


############################################################################
# Gibbs sampler
############################################################################

# PP is the vector of block/vertex sizes to be used.
# If block order is being inferred, PP should be sorted in descending order;
#   otherwise, it should be in arrival-order

K = size(PP,1)
N = sum(PP)
# pre-allocate sample arrays
gibbs_psi ? psi_gibbs = zeros(Float64,K,Int(ceil((n_iter-n_burn)/n_thin))) : nothing
gibbs_arrival_times ? T_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin))) : nothing
gibbs_alpha ? alpha_gibbs = zeros(Float64,Int(ceil((n_iter-n_burn)/n_thin))) : nothing
gibbs_ia_params ? ia_params_gibbs = zeros(Float64,n_ia_params,Int(ceil((n_iter-n_burn)/n_thin))) : nothing
gibbs_perm_order ? perm_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin))) : nothing

# initialize
gibbs_psi ? psi_current = 0.5*ones(Float64,K) : nothing

gibbs_alpha ? alpha_current = initialize_alpha(ntl_alpha_prior_dist) : alpha_current = [alpha_fixed]

if gibbs_ia_params
  arrival_params_current = initialize_arrival_params(ia_prior_params)
else
  arrival_params_current = arrival_params_fixed
end

if gibbs_arrival_times
  T_current = initialize_arrival_times(PP,alpha_current[1],ia_dist(arrival_params_current))
else
  T_current = T_data
end

perm_current = collect(1:K) # initial permutation order

# Gibbs sampler
ct_gibbs = 0 # counts when to store state of Markov Chain
for s in 1:n_iter
  gibbs_psi ? update_psi_parameters_partition!(psi_current,PP[perm_current],alpha_current[1]) : nothing
  gibbs_arrival_times ? update_arrival_times!(T_current,PP[perm_current],alpha_current[1],ia_dist(arrival_params_current)) : nothing
  gibbs_alpha ? update_ntl_alpha!(alpha_current,PP[perm_current],T_current,ntl_alpha_log_prior,w_alpha) : nothing
  gibbs_perm_order ? update_block_order!(perm_current,PP[perm_current],T_current,alpha_current[1]) : nothing
  gibbs_ia_params ? update_arrival_params!(arrival_params_current,T_current,N,ia_prior_params) : nothing
  # p_current = update_geometric_interarrival_param_partition(p_current,K,N,a,b)
  if (s > n_burn) && mod(s - n_burn,n_thin)==0
    ct_gibbs += 1
    gibbs_psi ? psi_gibbs[:,ct_gibbs] = psi_current : nothing
    gibbs_arrival_times ? T_gibbs[:,ct_gibbs] = T_current : nothing
    gibbs_alpha ? alpha_gibbs[ct_gibbs] = alpha_current[1] : nothing
    gibbs_ia_params ? ia_params_gibbs[:,ct_gibbs] = arrival_params_current : nothing
    gibbs_perm_order ? perm_gibbs[:,ct_gibbs] = perm_current : nothing
  end

end

############################################################################
# end
############################################################################


# psi_consistent = (PP_syn.-1)./cumsum(PP_syn)
# psi_mle = (PP_syn.-1)./(cumsum(PP_syn).-T_syn)
# psi_map = (PP_syn.-1-alpha)./(cumsum(PP_syn).-(1:K).*alpha.-2)
# psi_map[1] = 1
# psi_mean_gibbs = mean(psi_gibbs,2)
# plot(psi_map.-psi_mean_gibbs,lw=2)

##############################
# Gibbs sampler when observing an arbitrarily ordered partition (but not arrivals)
# ***** need to make this able to handle general interarrival distribution/updates
function gibbs_partition(n_iter::Int,n_burn::Int,n_thin::Int,
  PP_sorted::Vector{Int},alpha::Float64,ia_dist,a::Float64,b::Float64)

  K = size(PP_sorted,1)
  N = sum(PP_sorted)
  # pre-allocate
  psi_gibbs = zeros(Float64,K,Int(ceil((n_iter-n_burn)/n_thin)))
  T_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin)))
  sigma_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin)))
  p_gibbs = zeros(Float64,Int(ceil((n_iter-n_burn)/n_thin)))
  # initialize
  psi_current = 0.5*ones(Float64,K)
  p_current = rand(Beta(a,b))
  sigma_current = collect(1:K) # start in decreasing order
  T_current = initialize_arrival_times(PP_sorted[sigma_current],alpha,ia_dist(p_current))
  ct_gibbs = 0

  for n in 1:n_iter
    update_psi_parameters_partition!(psi_current,PP_sorted[sigma_current],alpha)
    update_block_order!(sigma_current,PP_sorted,T_current,alpha)
    update_arrival_times!(T_current,PP_sorted[sigma_current],alpha,ia_dist(p_current))
    p_current = update_geometric_interarrival_param_partition(p_current,K,N,a,b)
    if (n > n_burn) && mod(n - n_burn,n_thin)==0
      ct_gibbs += 1
      psi_gibbs[:,ct_gibbs] = psi_current
      sigma_gibbs[:,ct_gibbs] = sigma_current
      T_gibbs[:,ct_gibbs] = T_current
      p_gibbs[ct_gibbs] = p_current
    end

  end

  return psi_gibbs,T_gibbs,p_gibbs,sigma_gibbs


end

N = 2000
alpha = -10.
p = 0.25

# create intearrival distribution object and synthetic data
interarrival_dist = Geometric
Z_syn, PP_syn, T_syn = generateLabelSequence(N,alpha,interarrival_dist(p))

K = size(PP_syn,1)

n_gibbs = 20000
n_burn = 15000
n_thin = 1
a_beta = 1.
b_beta = 1.
PP_sorted = sort(PP_syn,rev=true)
@time psi_gibbs,T_gibbs,p_gibbs,sigma_gibbs =
  gibbs_partition(n_gibbs,n_burn,n_thin,PP_sorted,alpha,interarrival_dist,a_beta,b_beta)

#

#
# median_order = median(sigma_gibbs,2)
p_order = [0.05; 0.25; 0.5; 0.75; 0.95]
quantile_order = zeros(Float64,size(p_order,1),size(sigma_gibbs,1))
for q in 1:size(p_order,1)
  for k in 1:K
    quantile_order[q,k] = quantile(sigma_gibbs[k,:],p_order[q])
  end
end
scatter(PP_sorted,quantile_order',legend=false)
plot(quantile_order',legend=false)
