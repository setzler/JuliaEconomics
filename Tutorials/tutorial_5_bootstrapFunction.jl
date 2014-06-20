# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 5: Parallel Processing in Julia: Bootstrapping the MLE

srand(2)

using Distributions
using Optim


N=1000
K=10

genX = MvNormal(eye(K))
X = rand(genX,N)
X = X'
constant = ones(N)
X = [constant X]
genEpsilon = Normal(0, 1)
epsilon = rand(genEpsilon,N)
trueParams = [-K/2:K/2]*.02
Y = X*trueParams + epsilon
params0 = [trueParams,1]


function loglike(rho,x,y)
    beta = rho[1:K+1]
    sigma2 = exp(rho[K+2])
    residual = y-x*beta
    dist = Normal(0, sigma2)
    contributions = logpdf(dist,residual)
    loglikelihood = sum(contributions)
    return -loglikelihood
end

function bootstrapSamples(B)
    println("hi")
    M=convert(Int,N/2)
    samples = zeros(B,K+2)
    for b=1:B
        theIndex = randperm(N)[1:M]
        x = X[theIndex,:]
        y = Y[theIndex,:]
        function wrapLoglike(rho)
            return loglike(rho,x,y)
        end
        samples[b,:] = optimize(wrapLoglike,params0,method=:nelder_mead).minimum
    end
    samples[:,K+2] = exp(samples[:,K+2])
    println("bye")
    return samples
end
