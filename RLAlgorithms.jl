module RLAlgorithms
    include("RLEnvironments.jl")
    include("RLHelpers.jl")
    using Reexport
    @reexport using .RLHelpers
    @reexport using .RLEnvironments
    using StatsBase: sample, Weights
    include("battery.jl")

    export episode!, episodes!, test

# the episodes function will operate steps function for #steps times
# let steps be an integer times of 168, each episode is 168 steps

    function episodes!(f, memory, env, steps, maxt, FR_price_total, grid_price_total, signal_segment_total)
        Nt_FR_hour =  round(Int, 3600/2)
        j = 0
        done = false
        while j < steps  # each step means one day
            jj, done = episode!(f, memory, env, min(maxt, steps-j), FR_price_total[j+1:j+maxt], grid_price_total[j+1:j+maxt],
            signal_segment_total[j*Nt_FR_hour+1:(j+maxt)*Nt_FR_hour+1])
            j += jj
            done && break
        end
        return done
    end

    function episode!(f, memory,
        env, maxt, FR_price, grid_price, signal)
        done = false
        Nt_FR_hour =  round(Int, 3600/2)
        t = 0  # each t means each step (one step means one hour)
        while t < maxt # && !done
            FR_price_segment = FR_price[t+1]
            grid_price_segment = grid_price[t+1]
            signal_segment = signal[t*Nt_FR_hour+1:(t+1)*Nt_FR_hour+1]
            set!(env, FR_price_segment, grid_price_segment, signal_segment)
            s = copy(env.current)
            a = f(s)
            # step function has input (FR_price_segment, grid_price_segment, signal_segment)
            _, r1, r2, done = step!(env, a, FR_price_segment, grid_price_segment, signal_segment)
            append!(memory, s, a, r2, done)
            t += 1
            done && break
            # if done
            #     global retire_epoch += 1
            #     reset!(env)
            # end
        end
        return t,  done
    end
# here f is actor function
# the function episode!() operate for a step function for maxt steps  maxt here should be set as 168


    function test(f, env, FR_price_segment, grid_price_segment, signal_segment, maxt=28*24) where {S, R, A}
        reward1 = 0.0
        reward2 = 0.0
        state_copy = copy(env.state)
        current_copy = copy(env.current)
        s = reset!(env)
        Nt_FR_hour = Int(3600/2)
        for i in 1:maxt
            set!(env, FR_price_segment[i], grid_price_segment[i], signal_segment[(i-1)*Nt_FR_hour+1:(i)*Nt_FR_hour+1])
            s = env.current
            _, r1, r2, done = step!(env, f(s), FR_price_segment[i], grid_price_segment[i], signal_segment[(i-1)*Nt_FR_hour+1:(i)*Nt_FR_hour+1])  ## step function needs additional information
            reward1 += r1
            reward2 += r2
            done && break
        end
        env.state = state_copy
        env.current = current_copy
        return reward1, reward2
    end

end
