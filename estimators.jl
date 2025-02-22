function qzn_pr_estimator_1(n::Int, Kn::Real, alpha::Float64,
        a_prime_prior::Real, b_prime_prior::Real, S_n::Vector)

    # Expectation of Beta(1 + K_n, n -1 - K_n)
    a = (a_prime_prior + Kn - 1) / (a_prime_prior + b_prime_prior + n - 2)

    # Using K_n here is pointless- we renormalize away any denominator anyway
    unnormalized = max.((S_n - alpha),0)
    lp = log.(unnormalized ./ sum(unnormalized))

    log_qzn_pr = log(1-a) + lp
    log_qzn_pr_new = log(a)

    return log_qzn_pr, log_qzn_pr_new
end

function qzn_pr_estimator_MC(qz_prev:: Vector, nk_stats::Matrix, M::Int, n::Int, K_max::Int, alpha::Float64, a_prime::Real, b_prime::Real)
    # Monte Carlo estimate with M samples
    # nk_stats Matrix of size MxKmax with n_k
    qz_prev_samples = wsample(1:K_max, qz_prev, M) # z ~ \hat{q}_{n-1}(.)
    # update nk
    for m in 1:M
        nk_stats[m, qz_prev_samples[m]] += 1
    end
    Kprev = sum(nk_stats .!= 0, 2)
    qzn_pr = zeros(Float64, K_max + 1)
    p_new_cluster = (a_prime + Kprev - 1) ./ (a_prime + b_prime + n - 2)
    qzn_pr[K_max+1] = mean(p_new_cluster)
    for k in 1:K_max
        qzn_pr[k] = mean( (1 - p_new_cluster) .* max.(nk_stats[:,k] - alpha, 0) ./ (n - 1 - alpha.*Kprev) )
    end

    nk_stats, log.(qzn_pr[1:K_max]),  log(qzn_pr[K_max+1])
end

function qzn_pr_estimator_NRM(S_n::Vector, Sprod::Vector, Un_hat::Real, n::Int, K_max::Int, a::Real, tau::Real, sigma::Real)
    # Compute q^PR(zn) for the NRM mixture model cf Tank et al. 2014
    # a, tau and sigma are the NGGP's parameters

    log_qzn_pr_prevs = log.(max.(S_n - sigma, 0)) # cf Alg 1
    if sigma == 0 # DP
        return log_qzn_pr_prevs, log(a), 1
    end

    Kprev = (K_max - sum(Sprod)) # E[K_{n-1}] under \hat{q}(z_{1:n-1})

    # Compute argmax of q(U_{n-1}) by gradient ascent
    log_q = (U) -> -a/tau*(U + tau)^sigma + (n-1)*log(U) -(n - 1 - a*Kprev)*log(U + tau)
    grad_log_q = (U) -> -a/tau*(U + tau)^(sigma-1) + (n-1)/U -(n - 1 - a*Kprev)/(U + tau)

    last_log_q = log_q(Un_hat)
    step_size = 1.
    i = 1
    while true
        Un_hat += step_size * grad_log_q(Un_hat)
        new_log_q = log_q(Un_hat)
        println("obj[",i,"]: ", new_log_q)
        obj_diff = abs(new_log_q - last_log_q)
        ((obj_diff < 0.1) || i >= 100) && break
        last_log_q = new_log_q
        i += 1
    end

    log_qzn_pr_new = log(a) + sigma * log(Un_hat + tau)

    return log_qzn_pr_prevs, log_qzn_pr_new, Un_hat
end
