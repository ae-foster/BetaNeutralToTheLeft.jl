## evaluation metrics

function mean_arrival_time_Lp(T_inferred::Vector{Int},T_truth::Vector{Int},p::Real)
  """
  Returns mean (normalized by number of elements in `T_inferred`) L^`p` norm
    of difference between `T_inferred` and `T_truth`.
  """
  if size(T_inferred,1) != size(T_truth,1)
    error("T_inferred and T_truth must be the same length.")
  end
  K = size(T_inferred,1)
  if isfinite(p)
    d = (1/K)*sum( abs.((T_inferred .- T_truth).^p) ).^(1/p)
  else # L^∞-norm
    d = (1/K)*maximum( abs.(T_inferred .- T_truth) )
  end
  return d
end

function mean_arrival_time_Lp(T_inferred::Array{Int,2},T_truth::Vector{Int},p::Real)
  """
  Returns mean (normalized by number of columns of `T_inferred`) of
    mean (normalized by number of rows of `T_inferred`) L^`p` norm of
    differences between columns of `T_inferred` and `T_truth`.
  """
  s = zero(Float64)
  for j in 1:size(T_inferred,2)
    s += mean_arrival_time_Lp(T_inferred[:,j],T_truth,p)
  end
  return s./size(T_inferred,2)
end

function total_variation_distance(p::Vector{Float64},q::Vector{Float64})
  """
  Returns total variation distance between two discrete probabiliy distributions
    `p` and `q`. Elements of `p` and `q` are assumed to corresponding support points.
    If `size(p,1) != size(q,1)` then the shorter of the two is padded with zeros.
  """
  if abs(1.0 - sum(p)) > eps() || abs(1.0 - sum(q)) > eps()
    error("Elements of `p` or `q` do not sum to 1.")
  end
  np = size(p,1)
  nq = size(q,1)
  if np==nq
    d = 0.5*sum( abs.(p .- q) )
  elseif np > nq
    d = total_variation_distance(p,[q; zeros(Float64,np-nq)])
  else
    d = total_variation_distance([p; zeros(Float64,nq-np)],q)
  end
  return d
end
