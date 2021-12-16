# A. Number of node points
Ncp = 2 #number of concentration at the positive side
Ncn = 2 #number of concentration at the positive side
Nsei = 3
Ncum = 4
using JuMP
using Interpolations
using Sundials
using Ipopt
using DifferentialEquations

#=
B. Parameters
1: cathode, 2: anode
Dp,Dn: Solid phase diffusivity (m2/s),
kp, kn: Rate constant for lithium intercalation reaction (m2.5/(mol0.5s))
cspmax,csnmax: Maximum solid phase concentration at positive (mol/m3),
lp,ln : Region thickness (m),
ap,an: Particel surface area to volume (m2/m3)
Rpp ,Rpn: particle radius at positive (m),
ce: Electrolyte Concentration (mol/m3)
M[sei]: Molecular weight of SEI (Kg/mol)
Kappa[sei]: SEI ionic conductivity (S/m)
rho[sei]=: SEI density (Kg/m3)
ksei: rate constant of side reaction (C m/s mol)
F : Faraday's constant (C/mol) , R : Ideal gas constant (J/K/mol), T : Temperature (K)
=#

F=96487
R=8.3143
T=298.15
M_sei=0.073
Kappa_sei=5e-6
rho_sei=2.1e3
ksei=1.5e-12
Urefs = 0.4
Rsei = 0.01
area = 0.3108
cspmax=10350
csnmax=29480
lp=6.521e-5
lnn=2.885e-5
Rpp=1.637e-7
Rpn=3.596e-6
ce=1042
ep = 1-0.52
en = 1-0.619
ap=3*ep/Rpp
an=3*en/Rpn
kp=1.127e-7/F
kn=8.696e-7/F
Dn=8.256e-14
Dp=1.736e-14
TC = 2.3/area
Qmax = TC
P_nominal = TC*3.1
inner_state_dim = Ncp+Ncn+4+Nsei+Ncum
differential_vars = trues(Ncp+Ncn+4+Nsei+Ncum)
differential_vars[Ncp] = false
differential_vars[Ncp+Ncn] = false
differential_vars[Ncp+Ncn+1] = false
differential_vars[Ncp+Ncn+2] = false
differential_vars[Ncp+Ncn+3] = false
differential_vars[Ncp+Ncn+4] = false
differential_vars[Ncp+Ncn+5] = false
differential_vars[Ncp+Ncn+6] = false


nyears = 1
Totaltime = nyears*365*24*60*60                 # seconds
TotalHours = round(Int, Totaltime/3600)         # hours
dt_FR = 2                                       # time step for FR
Nt_FR_hour = round(Int, 3600/dt_FR)
Nt_FR_year = round(Int, 365*24*3600/dt_FR)
Nt_FR = round(Int, Totaltime/dt_FR)+1           # number of FR time step
TIME_FR = 0:dt_FR:Totaltime


# Files = ["FR/01_2017_Dynamic.csv", "FR/02_2017_Dynamic.csv", "FR/03_2017_Dynamic.csv", "FR/04_2017_Dynamic.csv",
#             "FR/05_2017_Dynamic.csv", "FR/06_2017_Dynamic.csv", "FR/07_2017_Dynamic.csv", "FR/08_2017_Dynamic.csv",
#             "FR/09_2017_Dynamic.csv", "FR/10_2017_Dynamic.csv", "FR/11_2017_Dynamic.csv", "FR/12_2017_Dynamic.csv"]

# global signal = []
# #cd("C:\\Users\\86156\\Desktop\\Battery_FP_UG-master\\Battery_FP_UG-master")
#
#
# for i in 1:12
#     filename = Files[i]
#     originalsignal = readdlm(filename, ',', Float64)[1:(end-1),:]   #positive means the grid sends power to battery (charging) and negative means grid buys power from battery(discharging).
#     if i == 1
#         global signal = vcat(originalsignal...)
#     else
#         global signal = [signal; vcat(originalsignal...)]
#     end
#     if i == 12
#        global signal = [signal; originalsignal[end,end]]
#     end
# end
# signal = min.(max.(signal, -1),1)
# #signal = repeat(signal; outer=[nyears])
# signal = signal[1:Nt_FR]
# FR_price = readdlm("FR/FR_Incentive.csv", ',', Float64)
# FR_price = vcat(FR_price'...)
# #FR_price = repeat(FR_price; outer=[nyears])
#
# grid_price = readdlm("FR/slow_Price.csv", ',', Float64)
# grid_price = vcat(grid_price'...)
#grid_price = repeat(grid_price; outer=[nyears])


function f_common(out,du,u,p,t)
    value = p[1]

    csp = u[1:Ncp]
    csn = u[Ncp+1:Ncp+Ncn]
    csp = max.(1, min.(csp, cspmax-1))
    csn = max.(1, min.(csn, csnmax-1))
    csp_avg = csp[1]
    csp_s = csp[2]
    csn_avg = csn[1]
    csn_s = csn[2]

    iint = u[Ncp+Ncn+1]
    phi_p = u[Ncp+Ncn+2]
    phi_n = u[Ncp+Ncn+3]
    pot = u[Ncp+Ncn+4]

    it = u[Ncp+Ncn+5]
    isei = u[Ncp+Ncn+6]
    delta_sei = u[Ncp+Ncn+7]


    cm = u[Ncp+Ncn+5+Nsei]
    cp = u[Ncp+Ncn+6+Nsei]
    Q = u[Ncp+Ncn+7+Nsei]
    cf = u[Ncp+Ncn+8+Nsei]


    #C2. Additional Equaions
    #Positive electrode
    theta_p = csp[Ncp]/cspmax
    Up = 7.49983 - 13.7758*theta_p.^0.5 + 21.7683*theta_p - 12.6985*theta_p.^1.5 + 0.0174967./theta_p -0.41649*theta_p.^(-0.5) -0.0161404*exp.(100*theta_p-97.1069) +
    	0.363031 * tanh.(5.89493 *theta_p -4.21921)
    jp = 2*kp*ce^(0.5)*(cspmax-csp[Ncp])^(0.5)*csp[Ncp]^(0.5)*sinh(0.5*F/R/T*(phi_p-Up))
    out[Ncp+Ncn+1] = (jp - (it/ap/F/lp))

    #Negative electrode
    theta_n = csn[Ncn]/csnmax
    Un = 9.99877 -9.99961*theta_n.^0.5 -9.98836*theta_n + 8.2024*theta_n.^1.5  + 0.23584./theta_n -2.03569*theta_n.^(-0.5) -
         1.47266*exp.(-1.14872*theta_n+2.13185)-
         9.9989*tanh.(0.60345*theta_n-1.58171)
    jn = 2*kn*ce^(0.5)*(csnmax-csn[Ncn])^(0.5)*csn[Ncn]^(0.5)*sinh(0.5*F/R/T*(phi_n-Un+(Rsei+delta_sei/Kappa_sei)*it/an/lnn));
    out[Ncp+Ncn+2] = (jn + (iint/an/F/lnn));
    out[Ncp+Ncn+3] = pot-phi_p + phi_n;
    #println(t, "   value   ",value, "     delta_sei  ", delta_sei,  "   it   ",it, "   isei   ",isei, "    phi_n  ", phi_n, "    ",  exp(-0.5*F/R/T*(phi_n - Urefs + it/an/lnn*(delta_sei/Kappa_sei+Rsei))) , "     ", exp(-0.5*F/R/T*(phi_n + it/an/lnn*(delta_sei/Kappa_sei))), "    ",it/an/lnn*(delta_sei/Kappa_sei)  )

    #C1. Governing Equations
    #Positive electrode
    out[1] = - 3*jp/Rpp - du[1]
    out[2] = 5*(csp_s - csp_avg) + Rpp*jp/Dp

    #Negative electrode
    out[Ncp+1] = - 3 * jn/Rpn - du[Ncp+1]
    out[Ncp+2] = 5*(csn_s - csn_avg) + Rpn*jn/Dn


    #C3. SEI layer Equations
    out[Ncp+Ncn+4] = -iint + it - isei
    out[Ncp+Ncn+5] = -isei+an*lnn*ksei*exp(-1*F/R/T*(phi_n - Urefs + it/an/lnn*(delta_sei/Kappa_sei+Rsei)))
    out[Ncp+Ncn+6] = isei*M_sei/F/rho_sei/an/lnn - du[Ncp+Ncn+7]   #d delta_sei/dt


    #C4. Charge stored
    out[Ncp+Ncn+4 + Nsei] = iint/3600 - du[Ncp+Ncn+5+Nsei];       #dcm/dt
    out[Ncp+Ncn+5 + Nsei] = it*pot/3600  - du[Ncp+Ncn+6+Nsei];       #dcp/dt
    out[Ncp+Ncn+6 + Nsei] = it/3600  - du[Ncp+Ncn+7+Nsei];         #dQ/dt
    out[Ncp+Ncn+7 + Nsei] = isei/3600  - du[Ncp+Ncn+8+Nsei];       #dcf/dt  isei = Cr
end



function  getinitial(csp_avg, csn_avg, delta_sei, value, mode)
    m = Model(with_optimizer(Ipopt.Optimizer, print_level = 0))

    theta_p_guess = min(0.9, csp_avg/cspmax)
    theta_n_guess = min(0.9, csn_avg/csnmax)
    Un_guess = 9.99877 -9.99961*theta_n_guess.^0.5 -9.98836*theta_n_guess + 8.2024*theta_n_guess.^1.5  + 0.23584./theta_n_guess -2.03569*theta_n_guess.^(-0.5) -
         1.47266*exp.(-1.14872*theta_n_guess+2.13185)-
         9.9989*tanh.(0.60345*theta_n_guess-1.58171)
    Up_guess = 7.49983 - 13.7758*theta_p_guess.^0.5 + 21.7683*theta_p_guess - 12.6985*theta_p_guess.^1.5 + 0.0174967./theta_p_guess -0.41649*theta_p_guess.^(-0.5) -
	0.0161404*exp.(100*theta_p_guess-97.1069) +
        0.363031 * tanh.(5.89493 *theta_p_guess -4.21921)

    if mode == 1
        it0 = value
    elseif mode == 2
        it0 = TC/2
    elseif mode == 3
	    pot0 = max(min(Up_guess-Un_guess, 3.3), 2.0)
        it0 = value/pot0
    end
    @variable(m, it, start= it0)
    @variable(m, iint, start= it0)

    @variable(m, csp_s, start = csp_avg)
    @variable(m, csn_s, start = csn_avg)

    @constraint(m,  5*(csp_s - csp_avg) + Rpp*it/F/Dp/ap/lp == 0)
    @constraint(m,  5*(csn_s - csn_avg) - Rpn*iint/F/Dn/an/lnn == 0)


    @variable(m, 0<=theta_p<=1, start=theta_p_guess)
    @variable(m, 0<=theta_n<=1, start= theta_n_guess)

    @constraint(m,  theta_p*cspmax == csp_s)
    @constraint(m,  theta_n*csnmax == csn_s)

    @variable(m, Up, start= Up_guess)
    @variable(m, phi_p, start= Up_guess)
    @variable(m, Un, start= Un_guess)
    @variable(m, phi_n, start= Un_guess)
    @NLconstraint(m,  (cspmax-csp_s)^(0.5)*csp_s^(0.5)*sinh(0.5*F/R/T*(phi_p-Up)) - (it/ap/F/lp/(2*kp*ce^(0.5))) == 0)
    @NLconstraint(m,  (csnmax-csn_s)^(0.5)*csn_s^(0.5)*sinh(0.5*F/R/T*(phi_n-Un+(delta_sei/Kappa_sei+Rsei)*it/an/lnn)) + iint/an/F/lnn/(2*kn*ce^(0.5)) == 0)

    @NLconstraint(m, Up == 7.49983 - 13.7758*theta_p^0.5 + 21.7683*theta_p - 12.6985*theta_p^1.5 + 0.0174967/theta_p -0.41649*theta_p^(-0.5) -
	0.0161404*exp(100*theta_p-97.1069) + 0.363031 * tanh(5.89493 *theta_p -4.21921))
    @NLconstraint(m, Un == 9.99877 -9.99961*theta_n^0.5 -9.98836*theta_n + 8.2024*theta_n^1.5  + 0.23584/theta_n -2.03569*theta_n^(-0.5) -
         1.47266*exp(-1.14872*theta_n+2.13185)- 9.9989*tanh(0.60345*theta_n-1.58171) )


    @variable(m, isei, start =  an*lnn*ksei*exp(-1*F/R/T*(Un_guess - Urefs + delta_sei/Kappa_sei*it0/an/lnn)))
    @NLconstraint(m, 1e4*(-isei+an*lnn*ksei*exp(-1*F/R/T*(phi_n - Urefs + (delta_sei/Kappa_sei+Rsei)*it/an/lnn))) == 0)
    @constraint(m, -iint + it -isei == 0)
    if mode == 1
        @constraint(m,  it == value)
    elseif mode == 2
        @constraint(m,  phi_p - phi_n == value)
    elseif mode == 3
        @constraint(m,  it*(phi_p-phi_n) == value)
    end
    JuMP.optimize!(m)
#    println("The initial solution status is ", JuMP.termination_status(m))
    iint0 = JuMP.value(iint)
    csp_s0 = JuMP.value(csp_s)
    csn_s0 = JuMP.value(csn_s)
    phi_p0 = JuMP.value(phi_p)
    phi_n0 = JuMP.value(phi_n)
    pot0 = phi_p0 - phi_n0
    it0 = JuMP.value(it)
    isei0 = JuMP.value(isei)
    return csp_s0, csn_s0, iint0, phi_p0, phi_n0, pot0, it0, isei0
end


V_max = 3.65
V_min = 2.0
maxC = 10
capacity_retire = 0.8
soc_min = 0.5
soc_max = 0.5
soc_min_stop = 0.1
soc_max_stop = 0.9

soc = 0.5
csp_avg0 = cspmax*1 - csnmax*soc*lnn*en/lp/ep            #49503.111
csn_avg0 = csnmax*soc
delta_sei0 = 1e-10
u0 = 0.1*ones(Ncp+Ncn+4 + Nsei+Ncum)
u0[1:Ncp] .= csp_avg0
u0[(Ncp+1):(Ncp+Ncn)] .= csn_avg0
u0[Ncp+Ncn+7] = delta_sei0              #delta_sei

u_start = copy(u0)
# u0[Ncp+Ncn+Nsei+5] = 0                  #cm
# u0[Ncp+Ncn+Nsei+6] = 0                  #cp
# u0[Ncp+Ncn+Nsei+7] = 0              	#Q
# u0[Ncp+Ncn+Nsei+8] = 0              	#cf


function simulation(TIME_FR_segment, signal_segment, FR_band, grid_band, u0)
	   TIME_FR_segment = 0:dt_FR:3600
	   P_FR_segment = signal_segment*FR_band .+ grid_band
           cf0 = u0[Ncp+Ncn+Nsei+8]
           fade = cf0/Qmax
           capacity_remain = 1 - fade
	   du0 = zeros(Ncp+Ncn+4+Nsei+Ncum)
	   tspan = (float(TIME_FR_segment[1]), float(TIME_FR_segment[end]))
       itp = interpolate((TIME_FR_segment,), P_FR_segment, Gridded(Linear()))
           function stop_cond(u,t,integrator)
               csn_avg = u[Ncp+1]
               soc_in = csn_avg/csnmax
               min_stop = capacity_remain*soc_min_stop
               max_stop = capacity_remain*soc_max_stop

               if t<=50
                   return (min_stop + max_stop)/2
               end

               if soc_in >= (min_stop + max_stop)/2
                   return (max_stop - soc_in)
               else
                   return (soc_in - min_stop)
               end
           end
           # affect!(integrator) = terminate!(integrator)
           # cb = ContinuousCallback(stop_cond,affect!, rootfind = true, interp_points=100)


           function f_FR(out,du,u,param,t)
                p_to_battery = itp[t]
                it = u[Ncp+Ncn+5]
                phi_p = u[Ncp+Ncn+2]
                phi_n = u[Ncp+Ncn+3]
                p = [p_to_battery]
                f_common(out,du,u,p,t)
                out[end] = (phi_p - phi_n)*it - p_to_battery  ## energy balance condition
            end

            ### modify algerbric variables
            csp_avg0 = u0[1]
            csn_avg0 = u0[Ncp+1]
            delta_sei0 = u0[Ncp+Ncn+7]
            csp_s0, csn_s0, iint0, phi_p0, phi_n0, pot0, it0, isei0 = getinitial(csp_avg0, csn_avg0, delta_sei0,  P_FR_segment[1], 3)
            u0[Ncp] = csp_s0
            u0[Ncp+Ncn] = csn_s0
            u0[Ncp+Ncn+1] = iint0                  #iint
            u0[Ncp+Ncn+2] = phi_p0                 #phi_p
            u0[Ncp+Ncn+3] = phi_n0                 #phi_n
            u0[Ncp+Ncn+4] = pot0                   #pot
            u0[Ncp+Ncn+5] = it0                     #it
            u0[Ncp+Ncn+6] = isei0                   #isei
            prob = DAEProblem(f_FR, du0, u0, tspan, differential_vars=differential_vars)
            sol = DifferentialEquations.solve(prob, IDA())
            csn_avg = (sol[end])[Ncp+1]
            soc_end = csn_avg/csnmax
            state_end = sol[end]
	    simulation_status = true
	    if sol.t[end] != tspan[end]
	        simulation_status = false
	    end
	    return soc_end, state_end, simulation_status  # solve the DAE problem, and return the soc_end = csn_avg/csnmax
end


nHours_Horizon = 1

function step_battery(FR_price_segment, grid_price_segment, signal_segment, FR_band, grid_band, u0)

    cf0 = u0[Ncp+Ncn+Nsei+8]
    fade = cf0/Qmax
    capacity_remain = 1 - fade
    csn_avg = u0[Ncp+1]
    soc = csn_avg/csnmax

    soc_stop = true
    TIME_FR_segment = 0:dt_FR:3600
    soc_end = copy(soc)
    while soc_stop
        soc_end, state_end, simulation_status = simulation(TIME_FR_segment, signal_segment, FR_band, grid_band, u0)
    # if stop prematurely
        if !simulation_status  || soc_end <= 0.1  || soc_end >= 0.9
#            println("soc too large or small")
            if FR_band >= P_nominal
                FR_band = FR_band - P_nominal/2
            else
                FR_band = FR_band/2
            end

            power_next = P_nominal*soc + FR_band*mean(signal_segment)
            if power_next <= P_nominal*capacity_remain*soc_min
                grid_band = P_nominal*capacity_remain*soc_min - power_next
            elseif power_next >= P_nominal*capacity_remain*soc_max
                grid_band = P_nominal*capacity_remain*soc_max -  power_next
            else
                grid_band = 0
            end
        else
            # FR_band_list[i_start_hour] = FR_band
            # buy_from_grid[i_start_hour] = grid_band
            u0 = state_end   # only the simulation process sucsess, then update the initial condition u0
            soc_stop = false
        end
    end

    cf_e = u0[Ncp+Ncn+Nsei+8]
    fade = cf_e/Qmax
    capacity_remain = 1 - fade
    done = (capacity_remain <= capacity_retire)
    expectedrevenue = 241407*P_nominal/1000
    reward1 = sum(FR_price_segment[h] * FR_band[h] - max(0,grid_band[h])*grid_price_segment[h] for h in 1:nHours_Horizon) # reward1 is the true profit
    reward2 = (sum(FR_price_segment[h] * FR_band[h] - max(0,grid_band[h])*grid_price_segment[h] for h in 1:nHours_Horizon)-5*(soc_end - 0.5*capacity_remain)^2 -
    expectedrevenue/(1-capacity_retire)/Qmax * (cf_e - cf0))
    #500 * (cf_e - cf0))
#    expectedrevenue/(1-capacity_retire)/Qmax * (cf_e - cf_0))
    return soc_end, capacity_remain, u0, reward1, reward2, done
end

function set_reset(FR_band, grid_band, signal_segment, u0)
    P_FR_segment = signal_segment*FR_band .+ grid_band
    csp_avg0 = u0[1]
    csn_avg0 = u0[Ncp+1]
    delta_sei0 = u0[Ncp+Ncn+7]
    csp_s0, csn_s0, iint0, phi_p0, phi_n0, pot0, it0, isei0 = getinitial(csp_avg0, csn_avg0, delta_sei0,  P_FR_segment[1], 3)
    u0[Ncp] = csp_s0
    u0[Ncp+Ncn] = csn_s0
    u0[Ncp+Ncn+1] = iint0                  #iint
    u0[Ncp+Ncn+2] = phi_p0                 #phi_p
    u0[Ncp+Ncn+3] = phi_n0                 #phi_n
    u0[Ncp+Ncn+4] = pot0                   #pot
    u0[Ncp+Ncn+5] = it0                     #it
    u0[Ncp+Ncn+6] = isei0                   #isei

    return u0
end
