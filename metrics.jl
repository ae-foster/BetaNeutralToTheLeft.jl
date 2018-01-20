function compare_nb_clusters(z, qz)
    # /!\ NOT TESTED YET
    true_nb_clusters = unique(z)
    approx_nb_cluster = size(qz, 2)

    return true_nb_clusters, approx_nb_cluster
end

function compute_clustering_quality(z, qz)
    # /!\ NOT TESTED YET
    # c_ij = 1 if i and j belongs to same cluster, 0 otherwise
    # c_approx_ij approx probability of belonging to the same cluster

    z = hash.(z)
    c = convert(Matrix{Int32}, z .== z')
    c_approx = qz' * qz

   return sum(sum(abs.(c - c_approx)))
end

# predictive log-likelihood ??
