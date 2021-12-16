#Pkg.activate("C:\\Users\\86156\\julia_bat_env")
#cd("C:\\Users\\86156\\Desktop\\Battery_FP_UG-master\\Battery_FP_UG-master")
using DifferentialEquations
using Statistics
using Sundials
using DelimitedFiles
#using PyPlot
using Interpolations
using Distributions
using JuMP
using Ipopt
using JLD
using GLPK
maxC = 10
include("setup1.jl")
include("data_analysis.jl")

nHours_Horizon = 1
# Total_hours =
area = 0.3108
TC = 2.3/area
Qmax = TC
P_nominal = TC*3.1
maxC = 10
# u = [FR_band, grid_band]
umax = [P_nominal*maxC, P_nominal*maxC]
umin = [0, -P_nominal*maxC]

function OptimalControl(u0, signal_segment, FR_price_segment, grid_price_segment, soc_min, soc_max)
    Total_time_horizon = nHours_Horizon*3600
    Nt_FR_Horizon = round(Int, Total_time_horizon/dt_FR)+1           		 # number of FR time step
    TIME_FR = 0:dt_FR:Total_time_horizon

    csp_avg0 = u0[1]
    csn_avg0 = u0[Ncp+1]
    soc0 = csn_avg0/csnmax
#    println("initial SOC", soc0)
    delta_sei0 = u0[Ncp+Ncn+7]
    cf0 = u0[Ncp+Ncn+Nsei+8]
    fade = cf0/Qmax
    capacity_remain = 1 - fade
    E_max = P_nominal
    E0 = E_max * soc0
    optimizer = GLPK.Optimizer
    m = Model(with_optimizer(optimizer))
    @variable(m,  umin[1] <= FR_band[1:nHours_Horizon] <= umax[1])
    @variable(m, umin[2] <= buy_from_grid[1:nHours_Horizon] <= umax[2])
    @variable(m, -P_nominal*maxC <= power[1:Nt_FR_Horizon] <= P_nominal*maxC)  # Nt_FR_horizon = total_time_horizon/dt_FR
    @variable(m, 0.1*(capacity_remain)*E_max <= energy[1:Nt_FR_Horizon+1] <= 0.9*(capacity_remain)*E_max)
    @variable(m, 0 <=buy_from_grid_plus[h in 1:nHours_Horizon] <= P_nominal*maxC)

    @constraint(m, energy[1] == E0)
    @constraint(m, [i in 1:Nt_FR_Horizon], energy[i+1] == energy[i] + dt_FR/3600*(power[i]))
    @constraint(m, [h in 1:nHours_Horizon], buy_from_grid_plus[h] >= buy_from_grid[h])
    @constraint(m, [i in 1:Nt_FR_Horizon], power[i] ==  signal_segment[i]*FR_band[max(1, Int(ceil(TIME_FR[i]/3600)))] + buy_from_grid[max(1, Int(ceil(TIME_FR[i]/3600)))])
    @constraint(m, energy[Nt_FR_Horizon+1] == (1-fade)*E_max*0.5)

    @objective(m, Min, -sum(FR_band[h]* FR_price_segment[h] for h in 1:nHours_Horizon)
    + sum(buy_from_grid_plus[h]*grid_price_segment[h] for h in 1:nHours_Horizon))

    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    FR_band = JuMP.value.(FR_band)[1]
    grid_band = JuMP.value.(buy_from_grid_plus)[1]
    return FR_band, grid_band, status, -JuMP.objective_value(m)
end


batch_num = 10
# for each batch, update the
function control_start(batch_num)
    for batch = 1: batch_num
       FR_price_sample = rand(d_FR_price, 364*24)
       FR_price_sample = min.(max.(FR_price_sample, min_FR_price), max_FR_price)

#       grid_price_sample = 1/5*vcat((rand(d_grid_price, 52)+rand(d_grid_price, 52)+rand(d_grid_price, 52)+rand(d_grid_price, 52)+rand(d_grid_price, 52))...)
       grid_price_sample = 1/50*vcat(sum(rand(d_grid_price,52) for i in 1:50)...)
       grid_price_sample = min.(max.(grid_price_sample, min_grid_price), max_grid_price)
       MPC(u_start,batch,FR_price_sample, grid_price_sample)
    end
end

control_start(batch_num)
