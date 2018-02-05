using Plots
using StatsBase

f = open("/data/flyrobin/foster/Documents/NTL.jl/sorted-mathoverflow.txt")
arrival_times = Dict{Any, Int}()
degrees = Dict{Any, Int}()
for (i, ln) in enumerate(eachline(f))
    a = split(ln)
    start = a[1]
    terminal = a[2]
    if ~haskey(arrival_times, start)
        arrival_times[start] = 2*i - 1
        degrees[start] = 1
    else
        degrees[start] += 1
    end
    if ~haskey(arrival_times, terminal)
        arrival_times[terminal] = 2*i
        degrees[terminal] = 1
    else
        degrees[terminal] += 1
    end
end

ts = sort(collect(values(arrival_times)))
degs = collect(values(degrees))

# Plots.plot(ts)
# Plots.gui()
#
# Plots.histogram(ts[2:end] - ts[1:end-1], bins=100)
# Plots.gui()

ef = ecdf(degs)
x = logspace(log(min(degs...)), log(max(degs...)), 100)
output = [(1-ef(t)) for t in x]
Plots.plot(x[output.>0], output[output.>0], xscale=:log10, yscale=:log10)
Plots.gui()

n = sum(degs)
xmin = 5
println("alpha hat ", 1 + n/(sum(log.(degs[degs.>xmin])) - n*log(xmin - 0.5)))
