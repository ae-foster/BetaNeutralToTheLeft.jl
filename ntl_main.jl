# Gibbs samplers for Beta NTL models of partitions, graphs, mixtures
using Distributions
using StatsBase
using Memoize

include("dataset.jl")
include("crp.jl")
include("ntl_gibbs.jl")
include("slice.jl")
include("evaluation.jl")


###########################################################################
# Gibbs sampler settings
###########################################################################
n_iter = 50000  # total number of Gibbs iterations to run
n_burn = 1000   # burn-in
n_thin = 100     # collect every `n_thin` samples

# set which components to update
gibbs_psi = true            # NTL Ψ paramters
gibbs_alpha = true          # NTL alpha parameter
gibbs_arrival_times = true  # arrival times
gibbs_ia_params = true     # arrival time distribution parameters
gibbs_perm_order = true   # order of blocks in partition/vertices in graph

## SET SEED
srand(0)

############################################################################
# Data
############################################################################
# need to add support for certain real data sets

base_dir = "/data/flyrobin/foster/Documents/NTL.jl/"
dataset_name = "dnc"

if dataset_name=="synthetic crp" # Synthetic data w/ CRP interarrivals
  println("Synethsizing data.")
  include("crp.jl")
  N = 1000
  ntl_alpha = 0.5
  crp_theta = 10.0
  crp_alpha = 0.6
  # create intearrival distribution object and synthetic data
  interarrival_dist = CRP(crp_theta,crp_alpha)
  Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist)

  gibbs_ia_params ? nothing : arrival_params_fixed = [crp_theta; crp_alpha]

  # function check
  assert(all(PP_data .== seq2part(Z_data)))
  assert(all(T_data .== get_arrivals(Z_data)))

elseif dataset_name=="synthetic geometric" # Synthetic data w/ geometric interarrivals
  println("Synethsizing data.")
  N = 1000
  ntl_alpha = 0.5
  geom_p = 0.25
  # create intearrival distribution object and synthetic data
  interarrival_dist = Geometric
  Z_data, PP_data, T_data = generateLabelSequence(N,ntl_alpha,interarrival_dist(geom_p))

  gibbs_ia_params ? nothing : arrival_params_fixed = [geom_p]

  # function check
  assert(all(PP_data .== seq2part(Z_data)))
  assert(all(T_data .== get_arrivals(Z_data)))

elseif dataset_name=="college msg"
  elist = readdlm("data/CollegeMsg.txt",Int64)
  Z_data = vec(elist[:,1:2]')
  PP_data = seq2part(Z_data)
  T_data = get_arrivals(Z_data)

elseif dataset_name=="dnc"
  elist = readdlm("data/dnc-temporalGraph/out.dnc-temporalGraph",Int64,skipstart=1)
  srt = vec(elist[sortperm(elist[:,4]),1:2]')
  uq = unique(srt) # preserves input order
  # relabel id's in time-sorted order
  Z_data = zeros(Int64,size(srt,1))
  for i in 1:size(uq,1)
    Z_data[srt .== uq[i]] = i
  end
  PP_data = seq2part(Z_data)
  T_data = get_arrivals(Z_data)

elseif startswith(dataset_name, "sorted-")
 # Assume that dataset_name is a filename
  PP_data, T_data = parseSnapData("$base_dir$dataset_name")

end

# sort partition if necessary
if gibbs_perm_order
  PP_sort = sortrows(hcat(PP_data,collect(1:size(PP_data,1))),rev=true)
  # sort ties in ascending order of original order for plotting purposes
  maxdeg = maximum(PP_sort)
  for j in 1:maxdeg
    PP_sort[PP_sort[:,1].==j,2] = sort(PP_sort[PP_sort[:,1].==j,2],rev=false)
  end
  perm_data = PP_sort[:,2]
  PP = PP_sort[:,1]
else
  PP = deepcopy(PP_data)
  perm_data = collect(1:size(PP,1))
end

K = size(PP,1)
N = sum(PP)
gibbs_alpha ? nothing : alpha_fixed = ntl_alpha

println("Finished pre-processing data.")

###########################################################################
# NTL alpha settings
###########################################################################

ntl_gamma_a = 1.
ntl_gamma_b = 1.

# prior is specified as a distribution on (0,Inf);
#   transformation to alpha ∈ (-Inf,1) will be performed during sampling as appropriate
ntl_alpha_prior_dist = Gamma(ntl_gamma_a,ntl_gamma_b)
ntl_alpha_log_prior = x -> logpdf(ntl_alpha_prior_dist,x)

w_alpha = 2.0 # slice sampling "window" parameter

###########################################################################
# Arrival time distribution settings
###########################################################################

arrivals = "crp"

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
  w_t = 2.0 # slice sampling w parameter for crp_theta
  w_a = 2.0 # slice sampling w parameter for crp_alpha

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
  # initialize_arrival_params = v -> [(K-1)/(N-1)]
  initialize_arrival_params = v -> [(K-1)/(N-1)]
  update_arrival_params! = update_geometric_interarrival_param!

end


############################################################################
# Gibbs sampler
############################################################################

# PP is the vector of block/vertex sizes to be used.
# If block order is being inferred, PP should be sorted in descending order;
#   otherwise, it should be in arrival-order

# pre-allocate sample arrays
println("Initializing sampler.")
gibbs_psi ? psi_gibbs = zeros(Float64,K,Int(ceil((n_iter-n_burn)/n_thin))) : psi_gibbs = []
gibbs_arrival_times ? T_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin))) : T_gibbs = []
gibbs_alpha ? alpha_gibbs = zeros(Float64,Int(ceil((n_iter-n_burn)/n_thin))) : alpha_gibbs = []
gibbs_ia_params ? ia_params_gibbs = zeros(Float64,n_ia_params,Int(ceil((n_iter-n_burn)/n_thin))) : ia_params_gibbs = []
gibbs_perm_order ? perm_gibbs = zeros(Int,K,Int(ceil((n_iter-n_burn)/n_thin))) : perm_gibbs = []

# initialize
gibbs_psi ? psi_current = 0.5*ones(Float64,K) : nothing
gibbs_alpha ? alpha_current = initialize_alpha(ntl_alpha_prior_dist) : alpha_current = [alpha_fixed]
if gibbs_ia_params
  arrival_params_current = initialize_arrival_params(ia_prior_params)
else
  arrival_params_current = arrival_params_fixed
end
if gibbs_arrival_times
  T_current = initialize_arrival_times(PP,alpha_current[1],Geometric((K-1)/(N-1)))
else
  T_current = T_data
end
perm_current = collect(1:K) # initial permutation order

# Gibbs sampler
n_print = 1000
ct_gibbs = 0 # counts when to store state of Markov Chain

t_elapsed = 0.
println("Running Gibbs sampler.")
tic();
for s in 1:n_iter
  gibbs_psi ? update_psi_parameters_partition!(psi_current,PP[perm_current],alpha_current[1]) : nothing ;
  gibbs_arrival_times ? update_arrival_times!(T_current,PP[perm_current],alpha_current[1],ia_dist(arrival_params_current)) : nothing ;
  gibbs_alpha ? update_ntl_alpha!(alpha_current,PP[perm_current],T_current,ntl_alpha_log_prior,w_alpha) : nothing ;
  gibbs_perm_order ? update_block_order!(perm_current,PP[perm_current],T_current,alpha_current[1]) : nothing ;
  gibbs_ia_params ? update_arrival_params!(arrival_params_current,T_current,N,ia_prior_params) : nothing ;
  # p_current = update_geometric_interarrival_param_partition(p_current,K,N,a,b)
  if (s > n_burn) && mod(s - n_burn,n_thin)==0
    ct_gibbs += 1 ;
    gibbs_psi ? psi_gibbs[:,ct_gibbs] = psi_current : nothing ;
    gibbs_arrival_times ? T_gibbs[:,ct_gibbs] = T_current : nothing ;
    gibbs_alpha ? alpha_gibbs[ct_gibbs] = alpha_current[1] : nothing ;
    gibbs_ia_params ? ia_params_gibbs[:,ct_gibbs] = arrival_params_current : nothing ;
    gibbs_perm_order ? perm_gibbs[:,ct_gibbs] = perm_current : nothing ;
  end
  if mod(s,n_print)==0
    t_elapsed += toq();
    println("Finished with ",s," samples out of ",n_iter,". Elapsed time is ",t_elapsed," seconds.")
    tic();
  end
end

println("Finished running Gibbs sampler.")

############################################################################
# end of Gibbs sampling code
############################################################################

############################################################################
# some performance evaluation metrics
############################################################################

# mean L^1 norm of difference between sampled arrivals and true arrivals
gibbs_arrival_times ? T_diff = mean_arrival_time_Lp(T_gibbs,T_data,1.0) : nothing


############################################################################
# some plots to check for parameter recovery
############################################################################
using Plots
gr()

plot(T_gibbs,legend=false)
plot!(T_data,linecolor="black",lw=2)

plot(alpha_gibbs,legend=false,lw=1.5)
hline!([ntl_alpha],line=(2,:dash,2.0,[:black]))

if arrivals=="crp"
  plot(ia_params_gibbs[1,:],legend=false,lw=1.5)
  hline!([crp_theta],line=(2,:dash,2.0,[:black]))
elseif arrivals=="geometric"
  plot(ia_params_gibbs[1,:],legend=false,lw=1.5)
  hline!([geom_p],line=(2,:dash,2.0,[:black]))
end

if arrivals=="crp"
  plot(ia_params_gibbs[2,:],legend=false,lw=1.5)
  hline!([crp_alpha],line=(2,:dash,2.0,[:black]))
end

plot(psi_gibbs,legend=false)

plot(PP[perm_data],legend=false)

# scatter(PP_data,psi_gibbs,legend=false)

# psi_consistent = (PP.-1)./cumsum(PP)
# psi_mle = (PP.-1)./(cumsum(PP).-T_data)
# psi_map = (PP.-1-ntl_alpha)./(cumsum(PP).-(1:K).*ntl_alpha.-2)
# psi_map[1] = 1
# psi_mean_gibbs = mean(psi_gibbs,2)
# plot(psi_mle.-psi_mean_gibbs,lw=2)
