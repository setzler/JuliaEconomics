#############################################################################
###### Lecture: Introduction to Structural Econometrics in Julia ############
###### 4. Numerical simulation of optimal agent behavior under constraints ##
###### Bradley Setzler, Department of Economics, University of Chicago ######
#############################################################################

####### Import Packages #########
import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import multiprocessing as mp

####### Prepare for Numerical Optimization #########

df = pd.read_csv("consump_leisure.csv")
N = df.shape[0]
numReps = 30 

####### Numerically Solve for Household Demand #########

def opt(g,t,w,e):
    def util(x):
        return -(g*np.log(x[0]) + (1-g)*np.log(x[1]))
    bnds = ((0.01, None), (0.01, 1))
    constr = ({'type': 'eq','fun' : lambda x: np.array([ (1.0-t)*(1.0-x[1])*w + e - x[0] ])})
    x0 = np.array([.5,.5])
    res = minimize(util, x0, constraints=constr, bounds=bnds, method='SLSQP', options={'disp': False})
    return res.x

####### Define Simulated Moments for Randomly Generated Epsilon #########

def sim_moments(params):
    g = params[0]
    t = params[1]
    wage = df['wage']
    #eps = df['epsilon']
    np.random.seed()                 # without this, eps is repeated identically on each processor
    eps = np.random.normal(size=N)
    demand = []
    for i in range(N):
        demand.append(opt(g,t,wage[i],eps[i]))
    demand = np.vstack(demand)
    c_moment = np.mean(demand[:,0]) - np.mean(df['consump'])
    l_moment = np.mean(demand[:,1]) - np.mean(df['leisure'])
    return np.array([c_moment,l_moment])

####### Define Sum of Squared Residuals in Parallel and Non-Parallel #########

def nonparallel_moments(params):
    params = np.exp(params)/(1.0+np.exp(params))  
    print(params)
    results = map(sim_moments,tuple(params for rep in range(numReps)))
    results = np.vstack(results)
    avg_c_moment = np.mean(results[:,0])
    avg_l_moment = np.mean(results[:,1])
    SSR = avg_c_moment**2 + avg_l_moment**2
    print(SSR)
    return SSR

def parallel_moments(params):
    params = np.exp(params)/(1.0+np.exp(params))  
    print(params)
    pool = mp.Pool(processes=4)           # the pool has to be regenerated on each run
    results = pool.map(sim_moments,tuple(params for rep in range(numReps)))
    results = np.vstack(results)
    avg_c_moment = np.mean(results[:,0])
    avg_l_moment = np.mean(results[:,1])
    SSR = avg_c_moment**2 + avg_l_moment**2
    print(SSR)
    return SSR

####### Minimize Sum of Squared Residuals in Parallel and Non-Parallel #########

def nonparallel_MSM():
  t0 = time.time()
  out = minimize(nonparallel_moments, [-1.,-1.], method='Nelder-Mead', options={'xtol':1e-6, 'ftol':1e-6}) 
  print(out)  
  t1 = time.time()
  totaltime = (t1-t0)/60.            
  return np.append(np.exp(out.x)/(1.0+np.exp(out.x)), totaltime)   
  
def parallel_MSM():
  t0 = time.time()
  out = minimize(parallel_moments, [-1.,-1.], method='Nelder-Mead', options={'xtol':1e-6, 'ftol':1e-6}) 
  print(out)  
  t1 = time.time()
  totaltime = (t1-t0)/60.            
  return np.append(np.exp(out.x)/(1.0+np.exp(out.x)), totaltime)   
                                                        
gamma_np_MSM, tau_np_MSM, time_np = nonparallel_MSM()  
gamma_p_MSM, tau_p_MSM, time_p = parallel_MSM()   

####### Compute Numerical Derivative of Simulated Demand for Consumption #########

def Dconsump_Dtau(gamma,tau,h):
    def consump_tau(tau):
        wage = df['wage']
        eps = df['epsilon']
        c_opt = np.empty((N,))
        for i in range(N):
            c_opt[i] = opt(gamma,tau,wage[i],eps[i])[0]
        return c_opt
    consump_diff = consump_tau(tau+h) - consump_tau(tau-h)
    return np.mean(consump_diff)/(2.*h)

#barpsi_MSM = Dconsump_Dtau(gamma_p_MSM,tau_p_MSM,.00001)

