## estimators for NTL parameters


function psi_mle(PP::Vector{Int},T::Vector{Int})
  """
  Returns maximum likelihood estimates of Ψ_j parameters
  """
  PP_p = cumsum(PP)
  return (PP .- 1)./(PP_p - T)
end

function psi_map(PP::Vector{Int},alpha::Float{64})
  """
  Returns maximum a posteriori estimates of Ψ_j parameters
  """
  PP_p = cumsum(PP)
  psi_hat = (PP .- 1 .- alpha)./(PP_p .- (1:size(PP,1)).*alpha .- 2)
  psi_hat[1] = 1
  return psi_hat
end

function psi_consistent(PP::Vector{Int})
  """
  Returns an asymptotically consistent estimator of Ψ_j parameters
  """
  return PP./cumsum(PP)
end
