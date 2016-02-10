# Bradley J. Setzler
# JuliaEconomics.com
# Tutorial 6: Kalman Filter for Panel Data and MLE in Julia
# Passed test on Julia 0.4, but is now much slower

using DataFrames
using Distributions
using Optim

function unpackParams(params)
    place = 0
    A = reshape(params[(place+1):(place+stateDim^2)],(stateDim,stateDim))
    place += stateDim^2
    V = diagm(exp(params[(place+1):(place+stateDim)]))
    place += stateDim
    C = zeros(stateDim*obsDim,stateDim)
    for j in [1:stateDim]
            C[(1+obsDim*(j-1)):obsDim*j,j] = [1,params[(place+1+(obsDim-1)*(j-1)):(place+(obsDim-1)*(j))]]
    end
    place += (obsDim-1)*stateDim
    W = diagm(exp(params[(place+1):(place+obsDim*stateDim)]))
    return ["A"=>A,"V"=>V,"C"=>C,"W"=>W]
end


function KalmanDGP(params)
    # initialize data
    data = zeros(N,stateDim*obsDim*T+1)
    # parameters
    unpacked = unpackParams(params)
    A = unpacked["A"]
    V = unpacked["V"]
    C = unpacked["C"]
    W = unpacked["W"]
    # draw from DGP
    for i=1:N
        # data of individual i
        iData = ones(stateDim*obsDim*T+1)
        current_state = rand(MvNormal(reshape(init_exp,(stateDim,)),init_var))
        iData[1:stateDim*obsDim] = rand(MvNormal(eye(obsDim*stateDim)))
        for t=2:T
            current_state = A*current_state + rand(MvNormal(V))
            iData[((t-1)*stateDim*obsDim+1):(t*stateDim*obsDim)] = C*current_state + rand(MvNormal(W))
        end
        # add individual to data
        data[i,:] = iData
    end
    return data
end


function incrementKF(params,post_exp,post_var,new_obs)
    # unpack parameters
    unpacked = unpackParams(params)
    A = unpacked["A"]
    V = unpacked["V"]
    C = unpacked["C"]
    W = unpacked["W"]
    # predict
    prior_exp = A*post_exp
    prior_var = A*post_var*A' + V
    obs_prior_exp = C*prior_exp
    obs_prior_var = C*prior_var*C' + W
    # update
    residual = new_obs - obs_prior_exp
    obs_prior_cov = prior_var*C'
    kalman_gain = obs_prior_cov*inv(obs_prior_var)
    post_exp = prior_exp + kalman_gain*residual
    post_var = prior_var - kalman_gain*obs_prior_cov'
    # step likelihood
    dist = MvNormal(reshape(obs_prior_exp,(length(obs_prior_exp),)),obs_prior_var)
    log_like = logpdf(dist,new_obs)
    return ["post_exp"=>post_exp,"post_var"=>post_var,"log_like"=>log_like]
end


function indivKF(params,init_exp,init_var,i)
    iData = data[i,:]
    # initialization
    post_exp = init_exp
    post_var = init_var
    init_obs = array(iData[obsDict[1]])'
    dist = MvNormal(eye(length(init_obs)))
    log_like=logpdf(dist,init_obs)
    for t = 1:(T-1)
        # predict and update
        new_obs = array(iData[obsDict[t+1]])'
        new_post = incrementKF(params,post_exp,post_var,new_obs)
        # replace
        post_exp = new_post["post_exp"]
        post_var = new_post["post_var"]
        # contribute
        log_like += new_post["log_like"]
    end
    return log_like
end


function sampleKF(params,init_exp,init_var)
    log_like = 0.0
    N = size(data,1)
    for i in 1:N
        log_like += indivKF(params,init_exp,init_var,i)
    end
    neg_avg_log_like = -log_like/N
    println("current average negative log-likelihood: ",neg_avg_log_like)
    return neg_avg_log_like[1]
end


function wrapLoglike(params)
    print("current parameters: ",params)
    return sampleKF(params,init_exp,init_var)
end



srand(2)
N = 1000
T = 4
stateDim = 2
obsDim = 3
init_exp = zeros(stateDim)
init_var = eye(stateDim)
params0 = [1.,0.,0.,1.,0.,0.,.5,-.5,.5,-.5,0.,0.,0.,0.,0.,0.]

data=KalmanDGP(params0)
data = DataFrame(data)
names!(data,[:one_1,:one_2,:one_3,:one_4,:one_5,:one_6,:two_1,:two_2,:two_3,:two_4,:two_5,:two_6,:three_1,:three_2,:three_3,:three_4,:three_5,:three_6,:four_1,:four_2,:four_3,:four_4,:four_5,:four_6,:outcome])
obsDict = {[:one_1,:one_2,:one_3,:one_4,:one_5,:one_6],[:two_1,:two_2,:two_3,:two_4,:two_5,:two_6],[:three_1,:three_2,:three_3,:three_4,:three_5,:three_6],[:four_1,:four_2,:four_3,:four_4,:four_5,:four_6]}

tic()
MLE = optimize(wrapLoglike,params0,method=:cg,ftol=1e-8)
optimParams = unpackParams(MLE.minimum)
toc()
