# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 5: Parallel Processing in Julia: Bootstrapping the MLE
# Passed test on Julia 0.4

using DataFrames
using Distributions
using Optim


params0 = [.1,.2,.3,.4,.5]
data = readtable("data.csv")
N = size(data,1)
Y = array(data[:Y])
X = array(data[[:one,:X1,:X2,:X3]])

function loglike(rho,y,x)
    beta = rho[1:4]
    sigma2 = exp(rho[5])+eps(Float64)
    residual = y-x*beta
    dist = Normal(0, sigma2)
    contributions = logpdf(dist,residual)
    loglikelihood = sum(contributions)
    return -loglikelihood
end

function bootstrapSamples(B)
    println("hi")
    samples = zeros(B,5)
    for b=1:B
		theIndex = sample(1:N,N)
		x = X[theIndex,:]
		y = Y[theIndex,:]
		function wrapLoglike(rho)
			return loglike(rho,y,x)
		end
		samples[b,:] = optimize(wrapLoglike,params0,method=:cg).minimum
	end
	samples[:,5] = exp(samples[:,5])
    println("bye")
    return samples
end
