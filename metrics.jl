function compare_nb_clusters(z, qz)
    true_nb_clusters = length(unique(z))
    approx_nb_cluster = size(qz, 2)

    return true_nb_clusters, approx_nb_cluster
end

function coclustering_l1(z, qz)
    # c_ij = 1 if i and j belongs to same cluster, 0 otherwise
    # c_approx_ij approx probability of belonging to the same cluster
    N = length(z)
    c = coclustering_matrix(z)
    c_approx = approx_coclustering_matrix(qz)
    return mean(abs.(c - c_approx))
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

function mutual_information(z, qz, S_n)
    K_true = max(z...)
    K_max = length(S_n)
    N = length(z)
    S_n_true = [count(z .== i) for i=1:K_true]
    mi = 0
    for k_true = 1:K_true
        for k_approx = 1:K_max
            p_bar = mean(qz[z .== k_true, k_approx])
            if p_bar > 0
                mi += (S_n_true[k_true]*p_bar/N) * log(p_bar*N/(S_n[k_approx]))
            end
        end
    end
    mi
end

function adjusted_rand_score(z, qz, S_n)
    N = length(z)
    S_n_true = [count(z .== i) for i=1:max(z...)]
    c = coclustering_matrix(z)
    c_approx = approx_coclustering_matrix(qz)
    ri = sum(c_approx[c .== 1]) + sum((1-c_approx)[c .== 0])
    p_same = sum(S_n.^2)/N
    q_same = sum(S_n_true.^2)/N
    eri = p_same*q_same + (N - p_same)*(N - q_same)
    mri = N^2
    return (ri - eri)/(mri - eri)
end
