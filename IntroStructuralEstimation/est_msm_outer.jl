#############################################################################
###### Lecture: Introduction to Structural Econometrics in Julia ############
###### 5. Parallelized estimation by the Method of Simulated Moments ########
###### Bradley Setzler, Department of Economics, University of Chicago ######
#############################################################################

####### Prepare for Parallelization #########
addprocs(3)                   # Adds 3 processors in parallel (the first is added by default)
print(nprocs())               # Now there are 4 active processors
require("est_msm_inner.jl")   # This distributes functions and data to all active processors

####### Define Sum of Squared Residuals in Parallel #########

function parallel_moments(params)
  params = exp(params)./(1.0+exp(params))   # rescale parameters to be in [0,1]
  results = @parallel (hcat) for i=1:numReps
    sim_moments(params)
  end
  avg_c_moment = mean(results[1,:])
  avg_l_moment = mean(results[2,:])
  SSR = avg_c_moment^2 + avg_l_moment^2
end

####### Minimize Sum of Squared Residuals in Parallel #########

using Optim
function MSM()
  out = optimize(parallel_moments,[0.,0.],method=:nelder_mead,ftol=1e-8)
  println(out)                                       # verify convergence
  exp(out.minimum)./(1.0+exp(out.minimum))           # return results in rescaled units
end

numReps = 12                                         # set number of times to simulate epsilon
gamma_MSM, tau_MSM = MSM()                           # Perform MSM

####### Compute Numerical Derivative of Simulated Demand for Consumption #########

function Dconsump_Dtau(g,t,h)
  opt_plus_h = hh_constrained_opt(g,t+h,array(df[:wage]),array(df[:epsilon]))
  opt_minus_h = hh_constrained_opt(g,t-h,array(df[:wage]),array(df[:epsilon]))
  (mean(opt_plus_h[:c_opt]) - mean(opt_minus_h[:c_opt]))/(2*h)
end

barpsi_MSM = Dconsump_Dtau(gamma_MSM,tau_MSM,.0001)
