function compare_nb_clusters(z, qz)
    true_nb_clusters = length(unique(z))
    approx_nb_cluster = size(qz, 2)

    return true_nb_clusters, approx_nb_cluster
end

function compute_clustering_quality(z, qz)
    # c_ij = 1 if i and j belongs to same cluster, 0 otherwise
    # c_approx_ij approx probability of belonging to the same cluster
    N = length(z)
    c = coclustering_matrix(z)
    c_approx = approx_coclustering_matrix(qz)
    return mean(abs.(c - c_approx)), mean(abs.(c - Diagonal(ones(Int, N))))
end

function coclustering_matrix(z)
    z_hashed = hash.(z)
    return convert(Matrix{Int32}, z_hashed .== z_hashed')
end

function approx_coclustering_matrix(qz)
    c_approx = qz * qz'
    c_approx += - Diagonal(c_approx) + eye(N)
    return c_approx
end
