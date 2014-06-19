# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 5: Parallel Processing in Julia: Bootstrapping the MLE
srand(2)
addprocs(4)
require("tutorial_5_bootstrapFunction.jl")
B=1000
b=250
samples_pmap = pmap(bootstrapSamples,[b,b,b,b])
samples = vcat(samples_pmap[1],samples_pmap[2],samples_pmap[3],samples_pmap[4])
# @elapsed pmap(bootstrapSamples,[b,b,b,b])
# @elapsed bootstrapSamples(B)
