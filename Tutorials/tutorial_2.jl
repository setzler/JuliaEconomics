# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 2: Maximum Likelihood Estimation (MLE) in Julia: The OLS Example

srand(2)

using Distributions
using Optim

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
trueParams = [0.1,0.5,-0.3,0.]
Y = X*trueParams + epsilon

function loglike(rho)
    beta = rho[1:4]
    sigma2 = exp(rho[5])
    residual = Y-X*beta
    dist = Normal(0, sigma2)
    contributions = logpdf(dist,residual)
    loglikelihood = sum(contributions)
    return -loglikelihood
end



params0 = [.1,.2,.3,.4,.5]
optimum = optimize(loglike,params0,method=:cg)
MLE = optimum.minimum
MLE[5] = exp(MLE[5])
println(MLE)


optimum = optimize(loglike,params0,method=:nelder_mead)
MLE = optimum.minimum
MLE[5] = exp(MLE[5])
println(MLE)
