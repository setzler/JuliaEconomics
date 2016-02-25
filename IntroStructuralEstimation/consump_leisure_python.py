
#############################################################################
###### Lecture: Introduction to Structural Econometrics in Julia ############
###### 3. Data generation, management, and regression visualization #########
###### Bradley Setzler, Department of Economics, University of Chicago ######
#############################################################################

####### Import Packages #########
import os
import numpy as np
import pandas as pd
from ggplot import *

####### Set Simulation Parameters #########
os.chdir("/Users/bradley/SpeedTest")
np.random.seed(123)           # set the seed to ensure reproducibility
N = 1000             # set number of agents in economy
gamma = .5           # set Cobb-Douglas relative preference for consumption
tau = .2             # set tax rate

####### Draw Income Data and Optimal Consumption and Leisure #########
epsilon = np.random.normal(size=N)                                               # draw unobserved non-labor income
wage = 10+np.random.normal(size=N)                                               # draw observed wage
consump = gamma*(1-tau)*wage + gamma*epsilon                     # Cobb-Douglas demand for c
leisure = (1.0-gamma) + ((1.0-gamma)*epsilon)/((1.0-tau)*wage)  # Cobb-Douglas demand for l

####### Organize, Describe, and Export Data #########
df = pd.DataFrame()
df['consump'] = consump
df['leisure'] = leisure
df['wage'] = wage
df['epsilon'] = epsilon
plot_c = ggplot(aes(x='wage',y='consump'),data=df) + stat_smooth()
ggsave(plot_c,"plot_c.svg")
df.to_csv("consump_leisure.csv", index=False)

