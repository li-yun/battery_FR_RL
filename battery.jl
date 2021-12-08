include("battery_setup.jl")
# include("RLEnvironments.jl")
# using .RLEnvironments
# using .RLEvironments: ContEnv
using Statistics
state_dim = 6
inner_state_dim = Ncp+Ncn+4+Nsei+Ncum
mutable struct FR_battery{S,A,R} #<: ContEnv{S,R,A}
    current::Vector{S}
    observationspace::ObservationSpace{S}
    actionspace::ActionSpace{A}
    state::Vector{S}

    function FR_battery{T}(state_dim, inner_state_dim) where T <: AbstractFloat
        observationspace = ObservationSpace{T}(state_dim)
        actionspace = ActionSpace([(-one(T), one(T)),(-one(T),one(T))])
        new{T, T, T}(zeros(T,state_dim), observationspace, actionspace, zeros(T,inner_state_dim))
    end
end

FR_battery() = FR_battery{Float64}()

# for the step function, the information of next step FR_price grid_price and signal should be given
# FR_band, grid_band
umax = [P_nominal*maxC, P_nominal*maxC]
umin = [0, -P_nominal*maxC]

function denormalizeU(u,ind=0)
    if ind==0
        return (umax .+ umin)/2 .+ (umax .- umin) ./ 2 .* u
    else
        return (umax[ind] .+ umin[ind])/2 .+ (umax[ind] .- umin[ind]) ./ 2 .* u[ind]
    end
end

function step!(env, action::Array{A,1}, FR_price_segment, grid_price_segment, signal_segment) where {S,A,R}
    u0 = env.state # action is a two element vector, namely FR_band, grid_band
    ctr = denormalizeU(action)
    FR_band = ctr[1]
    grid_band = ctr[2]
    soc_end, capacity_remain, u0, reward1, reward2, done = step_battery(FR_price_segment, grid_price_segment, signal_segment, FR_band, grid_band, u0)
#    bat_state = [Statistics.mean(signal_segment), Statistics.var(signal_segment),FR_price_segment[1],grid_price_segment[1],soc_end,capacity_remain]

#    @assert size(bat_state)[1] == state_dim
    env.state = u0
    env.current[5:6] .= soc_end, capacity_remain
    return env.current, reward1, reward2, done
end

function reset!(env; start = u_start)
    # this function should give an initialization of env.state and env.current
    env.state = start
    env.current = [0,0,0,0,0.5,1] # current is only used for generating control actions
end

### set the current state variables as the next step FR_signal and price
function set!(env, FR_price, grid_price, signal)
    env.current[1:4] .= Statistics.mean(signal), Statistics.var(signal), FR_price[1], grid_price[1]
end
