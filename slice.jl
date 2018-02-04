### Slice Sampling routine
### code taken from
### Michael Lindon A Functional Implementation of Slice Sampling in Julia
### https://michaellindon.github.io/julia/slice-sampling/


function slice_sampling(g,w::Float64,x::Float64)
    function lowerbound(L)
        g(L)<y ? L : lowerbound(L-w)
    end
    function upperbound(R)
        g(R)<y ? R : upperbound(R+w)
    end
    function shrinksample(L,R)
        z=rand(Uniform(L,R))
        if g(z)>y
            z
        elseif z>x
            shrinksample(L,z)
        else
            shrinksample(z,R)
        end
    end
    y=-1*rand(Exponential(1))+g(x)
    U=rand(Uniform(0,1))
    shrinksample(lowerbound(x-U*w),upperbound(x+(1-U)*w))
end


# gaussian mixture for testing

# function ss_test(n_iter::Int,n_burn::Int,n_thin::Int,logf::Function,w::Float64)
#   samples = zeros(Float64,Int(ceil((n_iter-n_burn)/n_thin)))
#   ct = 0
#   x = 0.0
#
#   for n in 1:n_iter
#     x = slice_sampling(logf,w,x)
#     if (n > n_burn) && mod(n - n_burn,n_thin)==0
#       ct += 1
#       samples[ct] = x
#     end
#   end
#   return samples
# end

### FOR TESTING PURPOSES
# mus = rand(Normal(0,10),3)
# sigmas = rand(Gamma(3,1),3)
# wts = rand(Uniform(0,1),3)
# wts = wts./sum(wts)
# gmixdist = MixtureModel([Normal(mus[1],sigmas[1]),Normal(mus[2],sigmas[2]),Normal(mus[3],sigmas[3])],wts)
#
# function gm(x::Float64)
#   return logpdf(gmixdist,x)
# end
#
# f = x->pdf(gmixdist,x)
# g = x->logpdf(gmixdist,x)
# plot(f,-20,20,legend=false)
#
# n_iter = 10000
# n_burn = 0
# n_thin = 1
# w = 1.0
# samples = ss_test(n_iter,n_burn,n_thin,g,w)


# sample_f = partial(slice_sampling,g,1.0)
# sample_f(0.0)
