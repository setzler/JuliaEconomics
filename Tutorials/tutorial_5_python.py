# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 5: Parallel Processing in Julia: Bootstrapping the MLE
# Python Comparison Script

import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.optimize import fmin
import multiprocessing as mp
import time

np.random.seed(2)
N=1000
K=10

X = np.random.multivariate_normal(np.zeros(K),np.eye(K),N)
constant = np.ones((N,1))
X = np.hstack((constant,X))
epsilon = np.random.normal(0,1,N)
trueParams = np.linspace(-K/2,K/2,K+1)*.02
Y = np.dot(X,trueParams) + epsilon
params0 = np.hstack((trueParams,1))


def loglike(rho,x,y):
    beta = rho[0:K+1]
    sigma2 = np.exp(rho[K+1])
    residual = y-np.dot(x,beta)
    contributions = mn.logpdf(residual,0,sigma2)
    loglikelihood = np.sum(contributions)
    return -loglikelihood

#def wrapLoglike(rho):
#    return loglike(rho,X,Y)
#out = fmin_cg(wrapLoglike,params0)

def bootstrapSamples(B):
    print "hi"
    M=N//2
    samples = np.zeros((B,K+2))
    for b in range(B):
        theIndex = np.random.permutation(N)[0:M]
        x = X[theIndex]
        y = Y[theIndex]
        def wrapLoglike(rho):
            return loglike(rho,x,y)
        samples[b,:] = fmin(wrapLoglike,params0)
    samples[:,K+1] = np.exp(samples[:,K+1])
    print "bye"
    return samples

B = 1000
b = 1000//4

time0 = time.time()
samples = bootstrapSamples(B)
    = time.time() - time0

time0 = time.time()
pool = mp.Pool(4)
time0 = time.time()
samples = pool.map(bootstrapSamples,[b,b,b,b])
time_4proc = time.time() - time0