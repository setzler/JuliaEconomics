#############################################################################
###### Lecture: Introduction to Structural Econometrics in Julia ############
###### 3. Data generation, management, and regression visualization #########
###### Bradley Setzler, Department of Economics, University of Chicago ######
#############################################################################

####### Set Simulation Parameters #########
cd("/Users/bradley/JuliaEconomics/IntroStructuralEstimation/")
srand(123)           # set the seed to ensure reproducibility
N = 1000             # set number of agents in economy
gamma = .5           # set Cobb-Douglas relative preference for consumption
tau = .2             # set tax rate

####### Draw Income Data and Optimal Consumption and Leisure #########
epsilon = randn(N)                                               # draw unobserved non-labor income
wage = 10+randn(N)                                               # draw observed wage
consump = gamma*(1-tau)*wage + gamma*epsilon                     # Cobb-Douglas demand for c
leisure = (1.0-gamma) + ((1.0-gamma)*epsilon)./((1.0-tau)*wage)  # Cobb-Douglas demand for l

####### Organize, Describe, and Export Data #########
using DataFrames
using Gadfly
df = DataFrame(consump=consump,leisure=leisure,wage=wage,epsilon=epsilon)  # create data frame
plot_c = plot(df,x=:wage,y=:consump,Geom.smooth(method=:loess))            # plot E[consump|wage] using Gadfly
draw(SVG("plot_c.svg", 4inch, 4inch), plot_c)                              # export plot as SVG
writetable("consump_leisure.csv",df)                                       # export data as CSV
