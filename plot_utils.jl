using Plots
include("metrics.jl")

plotly()

function z_qz_heatmap(z, qz)

    c = coclustering_matrix(z)
    c_approx = approx_coclustering_matrix(qz)
    sigma = sortperm(z)

    c = c[sigma,sigma]
    c_approx = c_approx[sigma,sigma]

    Plots.heatmap(c, aspect_ratio=:equal)
    Plots.gui()

    Plots.heatmap(c_approx, aspect_ratio=:equal)
    Plots.gui()

    Plots.heatmap(c - c_approx, aspect_ratio=:equal)
    Plots.gui()

end

function gaussian_scatters(X, z, qz)

    Plots.scatter(X[:, 1], X[:, 2], color=z, aspect_ratio=:equal)
    Plots.gui()

    z_approx = map(x->ind2sub(qz, x)[2], findmax(qz, 2)[2])
    Plots.scatter(X[:, 1], X[:, 2], color=z_approx, aspect_ratio=:equal)
    Plots.gui()

end

function pll_plot(tseries)

    Plots.plot(tseries[1, :])
    Plots.gui()

    Plots.plot(tseries[2, :])
    Plots.gui()

    Plots.scatter(tseries[1, :], tseries[2, :])
    Plots.gui()

end

function cluster_means_plot(emission, qtheta)
    if emission == "gaussian"
        Plots.plot(sum(abs2, qtheta[1], 1)')
        Plots.gui()
    end
end
