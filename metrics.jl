function compare_nb_clusters(z, qz)
    true_nb_clusters = length(unique(z))
    approx_nb_cluster = size(qz, 2)

    return true_nb_clusters, approx_nb_cluster
end

function compute_clustering_quality(z, qz)
    # c_ij = 1 if i and j belongs to same cluster, 0 otherwise
    # c_approx_ij approx probability of belonging to the same cluster
    N = length(z)
    z_hashed = hash.(z)
    c = convert(Matrix{Int32}, z_hashed .== z_hashed')
    c_approx = qz * qz'
    c_approx += - Diagonal(c_approx) + eye(N)
    return mean(abs.(c - c_approx)), mean(abs.(c - Diagonal(ones(Int, N))))
end

function loglikelihood(logp_emission, x, log_qzn, log_qzn_new, qtheta, theta_prior, K_max)
    log_qx = zeros(Float64, K_max+1)
    log_qx[K_max+1] = log_qzn_new + logp_emission(x, theta_prior...)
    log_qx[1:K_max] = log_qzn + reshape(logp_emission(x, qtheta...), K_max)

    offset = max.(log_qx)[1]

    return offset + log(sum(exp.(log_qx)))
end
