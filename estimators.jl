function qzn_pr_estimator_1(S_n::Vector, n::Int, alpha::Float64)
    unnormalized = max.((S_n - alpha),0)
    lp = log.(unnormalized ./ sum(unnormalized))
    return lp
end

function qzn_pr_estimator_2(S_n::Vector, Sprod::Vector, n::Int, K_max::Int, a::Float64, alpha::Float64)
    (1 - a) * max((S_n - alpha),0) / (n - 1 - alpha*(K_max - sum(Sprod)))
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
        qzn_pr[k] = mean( (1 - p_new_cluster) .* (nk_stats[:,k] - alpha) ./ (n - 1 - alpha.*Kprev) )
    end

    nk_stats, log.(qzn_pr[1:K_max]),  log(qzn_pr[K_max+1])
end

function qzn_pr_estimator_NRM(S_n::Vector, Sprod::Vector, n::Int, K_max::Int, a::Float64, tau::Float64, sigma::Float64)
    log_qzn_pr = zeros(Float64, K_max + 1)
    log_qzn_pr[1:K_max] = log(max.(S_n - sigma, 0))

    Kprev = (K_max - sum(Sprod))

    log_q = (U) -> -a/tau*(U + tau)^sigma + (n-1)*log(U) -(n - 1 - a*Kprev)*log(U + tau)
    grad_log_q = (U) -> -a/tau*(U + tau)^(sigma-1) + (n-1)/U -(n - 1 - a*Kprev)/(U + tau)
    Un_hat = 0
    last_log_q = log_q(Un_hat)
    step_size = 0.1
    while true
        Un_hat += step_size * grad_log_q(Un_hat)
        (abs(log_q(Un_hat) - last_log_q) < 0.01) && break
    end

    log_qzn_pr[K_max+1] = log(a) + sigma * log(Un_hat + tau)

    return log_qzn_pr[1:K_max], log_qzn_pr[K_max+1]
end
