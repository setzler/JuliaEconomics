# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 1: Introductory Example: Ordinary Least Squares (OLS) Estimation in Julia
# Passed test on Julia 0.4

srand(2)

using Distributions

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

function OLSestimator(y,x)
    estimate = inv(x'*x)*(x'*y)
    return estimate
end

estimates = OLSestimator(Y,X)
linreg_estimates = linreg(X_noconstant,Y)
