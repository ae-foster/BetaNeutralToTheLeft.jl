using Plots
using JLD
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

    z_approx = map(x->ind2sub(qz, x)[1], findmax(qz, 2)[2])
    Plots.scatter(X[:, 1], X[:, 2], color=z_approx, aspect_ratio=:equal)
    Plots.gui()

end
