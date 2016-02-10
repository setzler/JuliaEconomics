# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 5: Parallel Processing in Julia: Bootstrapping the MLE
# Passed test on Julia 0.4

srand(2)

using Distributions
using Optim
using DataFrames

N=1000
K=3

genX = MvNormal(eye(K))
X = rand(genX,N)
X = X'
X_noconstant = X
constant = ones(N)
X = [constant X]

genEpsilon = Normal(0, 1)
epsilon = rand(genEpsilon,N)
trueParams = [0.01,0.05,0.05,0.07]
Y = X*trueParams + epsilon

data = DataFrame(hcat(Y,X))
names!(data,[:Y,:one,:X1,:X2,:X3])
writetable("data.csv",data)

addprocs(3)
require("tutorial_5_bootstrapFunction.jl")
B=1000
b=250
samples_pmap = pmap(bootstrapSamples,[250,250,250,250])
samples = vcat(samples_pmap[1],samples_pmap[2],samples_pmap[3],samples_pmap[4])
# @elapsed pmap(bootstrapSamples,[b,b,b,b])
# @elapsed bootstrapSamples(B)
