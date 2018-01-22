function compare_nb_clusters(z, qz)
    true_nb_clusters = length(unique(z))
    approx_nb_cluster = size(qz, 2)

    return true_nb_clusters, approx_nb_cluster
end

function compute_clustering_quality(z, qz)
    # c_ij = 1 if i and j belongs to same cluster, 0 otherwise
    # c_approx_ij approx probability of belonging to the same cluster

    z_hashed = hash.(z)
    c = convert(Matrix{Int32}, z_hashed .== z_hashed')
    c_approx = qz * qz'

   return mean(abs.(c - c_approx))
end

# predictive log-likelihood ??
