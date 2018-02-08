using Distributions
using SpecialFunctions
using Plots
gr()

@inbounds function loglike(Tj, alpha::Real, Kn::Int, n::Int, ia::Distribution, nj::Array{Int}, nj_bar::Array{Int})
    """
    Tj: Array{Real}, arrival times (relaxed not to necesseraly be integers)
    alpha: Real, NTL parameter
    Kn: Int, number of clusters/partitions
    n: Int, number of datapoints
    ia: Distribution, Gamma Distribution of inter-arrival times
    nj: Array{Int}, degrees of clusters/nodes
    nj_bar: Array{Int}, cumulative sum of cluster degrees
    """
    b = 1 / ia.θ
    res = - lgamma(n - Kn * alpha)
    res += lgamma(Tj[1] - alpha) + lgamma(nj[1] - alpha) - lgamma(1 - alpha)
    @simd for j in 2:Kn
        res += lgamma(Tj[j] - j*alpha) + lgamma(nj[j] - alpha) - lgamma(Tj[j] - 1 - (j-1)*alpha) - lgamma(1 - alpha)
        res += ia.α*log(b) - lgamma(ia.α) + (ia.α-1)*log(Tj[j] - Tj[j-1])# - b*(Tj[j] - Tj[j-1])
        res += lgamma(nj_bar[j] - Tj[j] + 1) - lgamma(nj[j]) - lgamma(nj_bar[j-1] - Tj[j] + 2)
    end
    res += - b * Tj[Kn]
    res += log(1 - cdf(ia, n - Tj[Kn]))

    res
end

@inbounds function loglike_derivative_Tj(x::Real, j::Int, Tj, Kn::Int, alpha::Real, ia::Distribution, nj_bar::Array{Int})
    # Compute the derivative of the log joint with respect to Tj

    grad = digamma(x - j * alpha) # First terms of the product
    grad += - digamma(x - 1 - (j - 1) * alpha)

    grad += (ia.α - 1)/(x - Tj[j-1])
    if j < Kn grad += - (ia.α - 1)/(Tj[j+1] - x) end

    grad += -digamma(nj_bar[j] - x + 1) # terms from combinatorial coefficient
    grad += digamma(nj_bar[j-1] - x + 2)

    if j == Kn # Censoring term for last T_Kn
        grad += - b
        grad += pdf(ia, n - x) / (1 - cdf(ia, n - x)) # NOTE: right ?
        # grad = min(grad, 0) # NOTE: USEFULL ??
    end

    if (j == Kn) && isnan(pdf(ia, n - x) / (1 - cdf(ia, n - x)))
        println("---NaNs---")
        println("pdf(ia, n - x): ", pdf(ia, n - x))
        println("1 - cdf(ia, n - x): ", (1 - cdf(ia, n - x)))
    end

    grad
end

@inbounds function loglike_second_derivative_Tj(x::Real, j::Int, Tj, Kn::Int, alpha::Real, ia::Distribution, j_bar::Array{Int})
    # Compute the 2nd derivative of the log joint with respect to Tj

    hess = trigamma(x - j * alpha) # First terms of the product
    hess += - trigamma(x - 1 - (j - 1) * alpha)

    hess += -(ia.α - 1)/(x - Tj[j-1])^2
    if j < Kn hess += (ia.α - 1)/(Tj[j+1] - x)^2 end

    hess += trigamma(nj_bar[j] - x + 1) # terms from combinatorial coefficient
    hess += -trigamma(nj_bar[j-1] - x + 2)

    if (j == Kn)
        grad_pdf = ia.θ^(-ia.α)/gamma(ia.α) * (-(ia.α-1)*(n - x)^(ia.α-2) + 1/ia.θ*(n - x)^(ia.α-1)) * exp(-(n - x)/ia.θ)
        hess += (grad_pdf*(1 - cdf(ia, n - x)) - pdf(ia, n - x)^2) / (1 - cdf(ia, n - x))^2
    end
    hess
end


# function loglike_derivative_alpha(x::Real, Tj::Array{Real}, n::Int, Kn::Int, nj::Array{Int})
@inbounds function loglike_derivative_alpha(x::Real, Tj, n::Int, Kn::Int, nj::Array{Int})
    # Compute the derivative of the log joint with respect to alpha

    grad = Kn * digamma(n - Kn * x) # First term of the joint

    vec = zeros(Float64, Kn)
    @simd for j in 1:Kn
        vec[j] = -j*digamma(Tj[j] - j * x) - digamma(nj[j] - x)
        if j > 1 vec[j] += (j-1)*digamma(Tj[j] - 1 - (j - 1) * x) end
        vec[j] += digamma(1 - x)
    end
    grad += sum(vec)

    grad
end

@inbounds function loglike_second_derivative_alpha(x::Real, Tj, n::Int, Kn::Int, nj::Array{Int})
    # Compute the 2nd derivative of the log joint with respect to alpha
    hess = - Kn^2 * trigamma(n - Kn * x) # First term of the joint

    vec = zeros(Float64, Kn)
    @simd for j in 1:Kn
        vec[j] = j^2*trigamma(Tj[j] - j * x) + trigamma(nj[j] - x)
        if j > 1 vec[j] += (j-1)^2*trigamma(Tj[j] - 1 - (j - 1) * x) end
        vec[j] += - trigamma(1 - x)
    end
    hess += sum(vec)

    hess
end


# Load dataset
include("dataset.jl")
base_dir = "data/"
# dataset_name = "sorted-sx-superuser.txt"
dataset_name = "sorted-sx-mathoverflow.txt"
nj, T_data = parseSnapData("$base_dir$dataset_name")
Kn = length(nj)
n = sum(nj)
nj_bar = cumsum(nj)
println("n: ", n)
println("Kn: ", Kn)

# Distribution of inter-arrival time: Gamma
a = (n - 1) / (n - Kn)
b = (Kn - 1) / (n - Kn)
println("a: ", a)
println("b: ", b)
ia = Gamma(a, 1/b) # use scale 1/b and not rate b

# Initialise point estimates
alpha = -6.8
Tj = round.(cumsum((n / Kn / 1.05) * ones(Real, Kn)))
Tj[Kn] = n - Kn

# Optimisation hyperparameters
nb_epochs = 5000
batch_size = Int(round(Kn/10))
lr_Tj = 1000.
lr_alpha = 1000.
stochastic = false

@inbounds @simd for i in 1:nb_epochs
    if i % 100 == 0
        plot(1:Kn, Tj)
        plot!(1:Kn, T_data)
        savefig("plots/inferred_Tj_$i.pdf")
    end

    count = 0
    obj = loglike(floor.(Tj), alpha, Kn, n, ia, nj, nj_bar)
    println("obj ",i,": ", obj)
    # println("MSE Tj: ", sqrt(mean((floor.(Tj)-T_data).^2)))
    println("L1 Tj: ", mean(abs.(floor.(Tj)-T_data)))

    idx = stochastic ? rand(collect(2:Kn), batch_size) : 2:Kn
        @inbounds for j in idx # Step for Tj, j=2,...,Kn
            grad = loglike_derivative_Tj(Tj[j], j, Tj, Kn, alpha, ia, nj_bar)
            # second_der = loglike_second_derivative_Tj(Tj[j], j, Tj, Kn, alpha, ia, nj_bar)
            # println("grad: ", grad)
            # println("second_der: ", second_der)
            Tj[j] += lr_Tj * grad

            if Tj[j] < Tj[j-1]+1 Tj[j] = Tj[j-1]+1; count +=1  end ;#println("hard set below T[",j,"]") end
            if (j < Kn) && (Tj[j] > min(nj_bar[j-1]+1, Tj[j+1]-1)) Tj[j] = min(nj_bar[j-1]+1, Tj[j+1]-1); count +=1 end #; println("hard set above T[",j,"]")  end
            if (j == Kn) && (Tj[j] > min(nj_bar[j-1]+1, n)) Tj[j] = min(nj_bar[j-1]+1, n); count +=1 end #; println("hard set above T[",j,"]")  end
        end
    println("Prop saturated (%): ", round(count / Kn * 100, 2))

    # Step for alpha
    # grad = loglike_derivative_alpha(alpha, Tj, n, Kn, nj)
    # second_der = loglike_second_derivative_alpha(alpha, Tj, n, Kn, nj)
    # alpha += lr_alpha * (grad / second_der)
    # println("alpha: ", alpha)

end

plot(1:Kn, Tj)
plot!(1:Kn, T_data)
savefig("plots/inferred_Tj_final.pdf")
