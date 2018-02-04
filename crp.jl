##### Chinese Restaurant Process arrival time utilities

type CRPinterarrival <: DiscreteUnivariateDistribution
  theta::Float64
  alpha::Float64
  n::Int
  k::Int
  # crp::Bool
end

Base.minimum(s::CRPinterarrival) = 1
Base.maximum(s::CRPinterarrival) = Inf

function partial(f,a...)
  (b...) -> f(a...,b...)
end

function CRP(a...)
  """
  Utility function for passing arguments `n` and `k` during sampling updates
  """
  partial(CRPinterarrival,a...)
end

Distributions.logpdf(s::CRPinterarrival,x::Int64) = _logpdf(s,x)
function _logpdf(s::CRPinterarrival,x::Int64)
  ka = s.k*s.alpha
  nt = s.n + s.theta
  logp = log(s.theta + ka) - log(nt)
  if x > 1
    nka = s.n - ka
    for i in 2:x
      logp += log(nka + i - 2) - log(nt + i - 1)
    end
  end
  return logp
end

Distributions.pdf(s::CRPinterarrival,x::Int) = _pdf(s,x)
function _pdf(s::CRPinterarrival,x::Int)
  return exp.(logpdf(s,x))
end

Distributions.cdf(s::CRPinterarrival,x::Int64) = _cdf(s,x)
function _cdf(s::CRPinterarrival,x::Int64)
  ka = s.k*s.alpha
  nt = s.n + s.theta
  P1 = (s.theta + ka)/(nt)
  if x > 1
    nka = s.n - ka
    prev = P1
    P = 0
    for j in 2:x
      run = exp( log(prev) + log(nka + j - 2) - log(nt + j - 1) )
      P += run
      prev = run
    end
  end
  return x > 1 ? P1 + P : P1
end

Base.Random.rand(s::CRPinterarrival) = _rand(s)
function _rand(s::CRPinterarrival)
  coin = 0
  ct = 0
  while coin != 1
    ct += 1
    p = (s.theta + s.alpha*s.k)/(s.theta + s.n + ct - 1)
    coin = rand(Bernoulli(p))
  end
  return ct
end

# slice sampling utilities
function crp_theta_logpdf(theta::Float64,alpha::Float64,k::Int,n::Int,log_prior::Function)
  """
  calculate unnormalized log-pdf proportional to `theta` in the CRP

  log_prior is a function that returns the (possibly unnormalized) prior log-probability
    of `theta`
  """
  if theta <= -alpha
    error("Invalid theta! theta > -alpha must be satisfied.")
  end
  logp = log_prior(theta,alpha)
  for j in 1:(k-1)
    logp += log(theta + j*alpha)
  end

  for m in 1:(n-1)
    logp += -log(theta + m)
  end
  return logp
end

function crp_theta_trans_logpdf(theta_trans::Float64,alpha::Float64,k::Int,n::Int,log_prior::Function)
  """
  computes log-pdf when theta has been transformed to the entire real line

  theta_trans = log(theta + alpha) (for fixed alpha)
  """
  theta = exp(theta_trans) - alpha
  return crp_theta_logpdf(theta,alpha,k,n,log_prior) + theta_trans
end

function crp_alpha_logpdf(alpha::Float64,theta::Float64,T::Vector{Int},n::Int,log_prior::Function)
  """
  calculate unnormalized log-pdf proportional to `alpha` in the CRP

  log_prior is a function that returns the (possibly unnormalized) prior log-probability
    of `alpha`
  """
  if theta <= -alpha
    error("Invalid alpha: theta > -alpha must be satisfied.")
  end
  k = size(T,1)
  logp = log_prior(alpha)
  for j in 1:(k-1)
    logp += log(theta + j*alpha)
    delta = T[j+1] - T[j]
    for m in 1:delta
      logp += log(T[j] + m - 1 - j*alpha)
    end
  end
  return logp
end

function crp_alpha_trans_logpdf(alpha_trans::Float64,theta::Float64,T::Vector{Int},n::Int,log_prior::Function)
  """
  computes log-pdf when alpha has been transformed to the entire real line

  alpha_trans = log(alpha - max(0,-theta)) - log(1-alpha) for fixed theta
  """
  alpha = (exp(alpha_trans) + max(0,-theta))/(1 + exp(alpha_trans))
  return crp_alpha_logpdf(alpha,theta,T,n,log_prior) + alpha_trans - log(1 + exp(alpha_trans)) + log(1 - alpha)
end


# parameter updates
function update_crp_interarrival_params!(ia_params::Vector{Float64},T::Vector{Int},
  n::Int,log_prior_theta::Function,log_prior_alpha::Function,
  w_t::Float64,w_a::Float64)

  # update theta via slice sampling
  K = size(T,1)
  ss_gt = x -> crp_theta_trans_logpdf(x,ia_params[2],K,n,log_prior_theta)
  theta_trans = log(ia_params[1] + ia_params[2])
  theta_trans_ss = slice_sampling(ss_gt,w_t,theta_trans)
  ia_params[1] = exp(theta_trans_ss) - ia_params[2]

  ss_ga = x -> crp_alpha_trans_logpdf(x,ia_params[1],T,n,log_prior_alpha)
  alpha_trans = log(ia_params[2] - max(0.,-ia_params[1])) - log(1 - ia_params[2])
  alpha_trans_ss = slice_sampling(ss_ga,w_a,alpha_trans)
  ia_params[2] = (exp(alpha_trans_ss) + max(0,-ia_params[1]))/(1 + exp(alpha_trans_ss))

end

function initialize_crp_params(theta_prior::UnivariateDistribution,alpha_prior::UnivariateDistribution)
  alpha = rand(alpha_prior)
  theta = rand(theta_prior) - alpha
  return [theta; alpha]

end
