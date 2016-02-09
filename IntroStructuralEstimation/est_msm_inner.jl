#############################################################################
###### Lecture: Introduction to Structural Econometrics in Julia ############
###### 4. Numerical simulation of optimal agent behavior under constraints ##
###### Bradley Setzler, Department of Economics, University of Chicago ######
#############################################################################

####### Prepare for Numerical Optimization #########

using DataFrames
using JuMP
using Ipopt
df = readtable("consump_leisure.csv")
N = size(df)[1]

####### Numerically Solve for Household Demand #########

function hh_constrained_opt(g,t,w,e)
  m = Model(solver=IpoptSolver())                                         # define empty model solved by Ipopt algorithm
  @defVar(m, c[i=1:N] >= 0)                                               # define positive consumption for each agent
  @defVar(m, 0 <= l[i=1:N] <= 1)                                          # define leisure in [0,1] for each agent
  @addConstraint(m, c[i=1:N] .== (1.0-t)*(1.0-l[i]).*w[i] + e[i] )        # each agent must satisfy the budget constraint
  @setNLObjective(m, Max, sum{ g*log(c[i]) + (1-g)*log(l[i]) , i=1:N } )  # maximize the sum of utility across all agents
  status = solve(m)                                                       # run numerical optimization
  c_opt = getValue(c)                                                     # extract demand for c
  l_opt = getValue(l)                                                     # extract demand for l
  demand = DataFrame(c_opt=c_opt,l_opt=l_opt)                             # return demand as DataFrame
end

####### Define Simulated Moments for Randomly Generated Epsilon #########

function sim_moments(params)
  this_epsilon = array(df[:epsilon])                                          # randn(N)                                                     # draw random epsilon
  ggamma,ttau = params                                                        # extract gamma and tau from vector
  this_demand = hh_constrained_opt(ggamma,ttau,array(df[:wage]),this_epsilon) # obtain demand for c and l
  c_moment = mean( this_demand[:c_opt] ) - mean( df[:consump] )               # compute empirical moment for c
  l_moment = mean( this_demand[:l_opt] ) - mean( df[:leisure] )               # compute empirical moment for l
  [c_moment,l_moment]                                                         # return vector of moments
end
