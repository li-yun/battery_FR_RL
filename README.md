# battery_FR_RL
scripts for the battery_FR_RL (code scripts for reproducing the simulation of the paper entitled "Deep Learning-based Predictive Control of Battery Management for Frequency Regulation")  
RL_process directory contains all code scripts for learning the FR policy via ddpg (initializations of the actor and the critic are provided).  
FR directory contains the data of FR signals, FR prices, and electricity prices.  
LF_MPC directory contrains the code scripts for simulating the LF-MPC policy.  
DNN_train directory contrains the code scripts for conducting the "SL process" (train initializations of the actor and the critic).  
generate_training_data directory contrains the code scripts for generating the training data set for conducting the "SL process".  
Code scrpits for implementing the "HF-MPC" are given in "https://github.com/zavalab/JuliaBox/ tree/master/Battery_FP"
