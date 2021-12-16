# Pkg.activate("C:\\Users\\86156\\batt_env")

using DifferentialEquations
using Statistics
using Sundials
using DelimitedFiles
using Interpolations
using Distributions
using JuMP
using Ipopt
using JLD
using GLPK
maxC = 10
include("setup1.jl")

umax = [P_nominal*maxC, P_nominal*maxC]
umin = [0, -P_nominal*maxC]

nHours_Horizon = 1
# Total_hours =

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
    @variable(m,  0 <= FR_band[1:nHours_Horizon] <= maxC*P_nominal)
    @variable(m, -P_nominal*maxC <= buy_from_grid[1:nHours_Horizon] <= P_nominal*maxC)
    @variable(m, -P_nominal*maxC <= power[1:Nt_FR_Horizon] <= P_nominal*maxC)  # Nt_FR_horizon = total_time_horizon/dt_FR
    @variable(m, 0.1*(capacity_remain)*E_max <= energy[1:Nt_FR_Horizon+1] <= 0.9*(capacity_remain)*E_max)
    @variable(m, 0 <=buy_from_grid_plus[h in 1:nHours_Horizon] <= P_nominal*maxC)

    @constraint(m, energy[1] == E0)
    @constraint(m, [i in 1:Nt_FR_Horizon], energy[i+1] == energy[i] + dt_FR/3600*(power[i]))
    @constraint(m, [h in 1:nHours_Horizon], buy_from_grid_plus[h] >= buy_from_grid[h])
    @constraint(m, [i in 1:Nt_FR_Horizon], power[i] ==  signal_segment[i]*FR_band[max(1, Int(ceil(TIME_FR[i]/3600)))] + buy_from_grid[max(1, Int(ceil(TIME_FR[i]/3600)))])
    @constraint(m, energy[Nt_FR_Horizon+1] == E_max*0.5*capacity_remain)

    @objective(m, Min, -sum(FR_band[h]* FR_price_segment[h] for h in 1:nHours_Horizon)
    + sum(buy_from_grid_plus[h]*grid_price_segment[h] for h in 1:nHours_Horizon))

    JuMP.optimize!(m)
    status = JuMP.termination_status(m)
    # println("The optimization status is", JuMP.termination_status(m))
    FR_band = JuMP.value.(FR_band)[1]
    grid_band = JuMP.value.(buy_from_grid_plus)[1]
    return FR_band, grid_band, status
    # if status == :Optimal
    #     println("Optimization Problem is solved")
    #     println("FR_band: ", FR_band, "grid_band: ",  grid_band)
    #     return FR_band, grid_band, status
    # else
    #     println("Optimization Problem is unsolved")
    # end
end

MPC(u0)
