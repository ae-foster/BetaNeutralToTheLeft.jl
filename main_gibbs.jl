# Gibbs samplers for Beta NTL models of partitions, graphs, mixtures
using Distributions
using StatsBase

include("estimators.jl")
include("dataset.jl")
include("ntl_gibbs.jl")


##### generate synthetic label sequence

# set number of labels; parameters
N = 1000
alpha = 0.5
p = 0.25

# create intearrival distribution object and synthetic data
interarrival_dist = Geometric
Z_syn, PP_syn, T_syn = generateLabelSequence(N,alpha,interarrival_dist(p))

# function check
assert(all(PP_syn .== seq2part(Z_syn)))
assert(all(T_syn .== get_arrivals(Z_syn)))

##############################
# Gibbs sampler when observing the label sequence
n_gibbs = 10000
n_burn = 1000
n_thin = 100
K = size(T_syn,1)

function gibbs_sequence(n_iter::Int,n_burn::Int,n_thin::Int,K::Int,Z::Vector{Int},alpha::Float64)

  psi_gibbs = zeros(Float64,K,(n_iter-n_burn)/n_thin)
  psi_current = 0.5*ones(Float64,K)
  ct_gibbs = 0
  PP = seq2part(Z)


  for n in 1:n_iter

    update_psi_parameters_partition!(psi_current,PP,alpha)
    # update_psi_parameters_sequence!(psi_current,Z_syn,alpha)
    if (n > n_burn) && mod(n - n_burn,n_thin)==0
      ct_gibbs += 1
      psi_gibbs[:,ct_gibbs] = psi_current
    end


  end
  return psi_gibbs
end

psi_gibbs = gibbs_sequence(n_gibbs,n_burn,n_thin,K,Z_syn)

# using Plots
# gr()
# plot(psi_gibbs[2,:])
# plot(mean(psi_gibbs,2)[25:50],lw=3)
# plot(PP_syn[1:50],lw=2)
#
# psi_consistent = (PP_syn.-1)./cumsum(PP_syn)
# psi_mle = (PP_syn.-1)./(cumsum(PP_syn).-T_syn)
# psi_map = (PP_syn.-1-alpha)./(cumsum(PP_syn).-(1:K)*alpha.-2)
# psi_map[1] = 1
# psi_mean_gibbs = mean(psi_gibbs,2)
#
# plot_idx = 25:50
# plot(plot_idx,[psi_consistent[plot_idx],psi_mle[plot_idx],psi_map[plot_idx],psi_mean_gibbs[plot_idx]])
#
# plot(psi_map.-psi_mean_gibbs,lw=2)


##############################
# Gibbs sampler when observing arrival-ordered partition (but not arrivals)
# ***** need to make this able to handle general interarrival distribution/updates
function gibbs_ordered_partition(n_iter::Int,n_burn::Int,n_thin::Int,
  PP::Vector{Int},alpha::Float64,ia_dist,a::Float64,b::Float64)

  K = size(PP,1)
  N = sum(PP)
  # initialize
  psi_gibbs = zeros(Float64,K,Int(ceil((n_iter-n_burn)/n_thin)))
  T_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin)))
  p_gibbs = zeros(Float64,Int(ceil((n_iter-n_burn)/n_thin)))
  psi_current = 0.5*ones(Float64,K)
  p_current = rand(Beta(a,b))
  T_current = initialize_arrival_times(PP,alpha,ia_dist(p_current))
  ct_gibbs = 0

  for n in 1:n_iter
    update_psi_parameters_partition!(psi_current,PP,alpha)
    update_arrival_times!(T_current,PP,alpha,ia_dist(p_current))
    p_current = update_geometric_interarrival_param_partition(p_current,K,N,a,b)
    if (n > n_burn) && mod(n - n_burn,n_thin)==0
      ct_gibbs += 1
      psi_gibbs[:,ct_gibbs] = psi_current
      T_gibbs[:,ct_gibbs] = T_current
      p_gibbs[ct_gibbs] = p_current
    end

  end

  return psi_gibbs,T_gibbs,p_gibbs
end


n_gibbs = 20000
n_burn = 5000
n_thin = 100
a_beta = 1.
b_beta = 1.
psi_gibbs,T_gibbs,p_gibbs = gibbs_ordered_partition(n_gibbs,n_burn,n_thin,PP_syn,alpha,interarrival_dist,a_beta,b_beta)


# psi_consistent = (PP_syn.-1)./cumsum(PP_syn)
# psi_mle = (PP_syn.-1)./(cumsum(PP_syn).-T_syn)
# psi_map = (PP_syn.-1-alpha)./(cumsum(PP_syn).-(1:K).*alpha.-2)
# psi_map[1] = 1
# psi_mean_gibbs = mean(psi_gibbs,2)
# plot(psi_map.-psi_mean_gibbs,lw=2)

# ***** need to test permutation inference *****


##############################
# generate synthetic d-dimensional diagonal Gaussian mixture data
# treat variance as known
prior_mu = 0.
prior_sigma = 100.
# cluster_mu = 0
cluster_sigma = 1.
d = 2
# prior_mu = [0; 0]
# prior_sigma = 10*eye(Float64,d,d)
# cluster_mu = [0; 0]
# cluster_sigma = 1*eye(Float64,d,d)

K = size(T_syn,1)
theta_syn = zeros(Float64,d,K)
mu = prior_mu*ones(Float64,d)
delta_theta_syn = rand(MvNormal(ones(Float64,d),eye(Float64,d,d)),K-1)
sigma = prior_sigma*eye(Float64,d,d)
for j in 1:K
  theta_syn[:,j] = rand(MvNormal(mu,sigma))
  mu += delta_theta_syn[j]
end

scatter(theta_syn[1,:],theta_syn[2,:],legend=false)

X_syn = zeros(Float64,d,size(Z_syn,1))
c_sigma = cluster_sigma*eye(Float64,d,d)
for n in 1:size(X_syn,2)
  X_syn[:,n] = rand(MvNormal(theta_syn[:,Z_syn[n]],c_sigma))
end
scatter(X_syn[1,:],X_syn[2,:],color=Z_syn,legend=false)
scatter!(theta_syn[1,:],theta_syn[2,:],legend=false)

##############################
# Gibbs sampler for Gaussian mixture data
n_gibbs = 1000
n_burn = 100
n_thin = 1

function gibbs_gaussian_mixture(n_iter::Int,n_burn::Int,n_thin::Int,X::Array{Real},alpha::Float64,
  ia_dist,a::Float64,b::Float64)
  """
  - `n_iter`: number of sampler iterations to run
  - `n_burn`: number of initial iterations to discard
  - `n_thin`: thinning factor (e.g. set to 10 to collect every 10th sample update)
  - `X`: data array
  - `alpha`: 'discount' parameter for clustering
  - `ia_dist`: interarrival distribution (without parameters)
  *** CURRENTLY ONLY SUPPORTS Geometric
  - `a`,`b`: hyperparameters for Beta(a,b) prior on Geometric parameter
  """

  # initialize

end
