# cd("C:\\Users\\86156\\Desktop\\CC_battery\\FR_reinforce\\RL_init")
# Pkg.activate("C:\\Users\\86156\\julia_bat_env_new")
using Pkg
Pkg.activate("/project/6033094/yunliubc/rl_fr/julia_package")

using Flux
using Flux: Optimise.update!
using Statistics
using Statistics: mean
using DifferentialEquations
using Sundials
using DelimitedFiles
using Interpolations
using Distributions
using JuMP
using Ipopt
using JLD
using BSON

include("RLAlgorithms.jl")  # for defining episode function
include("data_analysis.jl")
# include("RLHelpers.jl")
using .RLAlgorithms
# using .RLHelpers
include("battery.jl")

include("battery_setup.jl")
                                    # time step for FR
Nt_FR_hour = round(Int, 3600/2)
# using JLD
# using GLPK
# include("data_analysis.jl")  # used for generating training data
#include("battery_setup.jl")  # used for defining step function and reset function
# include("battery.jl") # provide the definition of struct FR_battery
state_dim = 6
inner_state_dim = Ncp+Ncn+4+Nsei+Ncum

test_horizon = 7*24 # hours
FR_price_test = FR_price_orig[1:test_horizon]
grid_price_test = grid_price_orig[1:test_horizon]
signal_test = signal[1:test_horizon*Nt_FR_hour+1]
epoch_week = 1  # one week corresponds to 168 hours


function act(actor, s, noisescale)
    a = actor(s)
    return max.(-1, min.(1, a .+ noisescale .* randn(Float64,size(a))))
#    return clamp.(a .+ noisescale .* randn(Float32, size(a)), -1f0, 1f0)
end

function run!(actor, aopt, critic, copt, env;
        epochs=50, steps=168*epoch_week, maxt=168, batchsize=160, noisescale=5f-2, γ=90f-2, τ=1f-2) ## steps  = 168*4
    rewards1 = zeros(Float64, epochs)
    rewards2 = zeros(Float64, epochs)
    actortar = deepcopy(actor)
    critictar = deepcopy(critic)
    memory = ReplayMemory{Float64, Float64, Float64}(
        length(env.observationspace), length(env.actionspace), 10*steps, batchsize
    )
    actorps = Flux.params(actor)
    criticps = Flux.params(critic)
# requires price and signal data
# for each epoch, a week data is provided 24*7 = 168
    reset!(env)

#    global retire_epoch = 0

    for i in 1:epochs

        # if retire_epoch > 5
        #     break
        # end

        FR_price_sample = rand(d_FR_price, 7*epoch_week*24)  # One epoch is corresponding to epoch_week weeks data
        FR_price_sample = min.(max.(FR_price_sample, min_FR_price), max_FR_price)

        grid_price_sample = (1/50)*vcat( sum(rand(d_grid_price, epoch_week) for i in 1:50)...) #4
#        grid_price_sample = min.(max.(grid_price_sample, min_grid_price), max_grid_price)

        signal_start = rand(signal_index)  # maximum signal index should be less than certain number
        signal_end = signal_start+Nt_FR_hour*steps+1
        signal_segment = signal[signal_start:signal_end]
        done = episodes!(memory, env, steps, maxt, FR_price_sample, grid_price_sample, signal_segment) do s
            act(actor, s, noisescale)
        end

        done && break

        for _ in 1:steps*4   ## steps is the number of steps for updating the actor and critic
            s1, a1, r, s2, done = sample(memory)

            atar2 = actortar(s2)
            Qtars2 = vec(critictar(vcat(s2, atar2)))
            Qtar = r .+ (1 .- done) .* γ .* Qtars2
            cgs = Flux.gradient(criticps) do
                Q = vec(critic(vcat(s1, a1)))
                return Flux.mae(Q, Qtar)
            end
            update!(copt, criticps, cgs)

            ags = Flux.gradient(actorps) do
                as = actor(s1)
                return -mean(critic(vcat(s1, as)))
            end
            update!(aopt, actorps, ags)

            softupdate!(actortar, actor, τ)
            softupdate!(critictar, critic, τ)
        end
        println("finished epoch: ", i)
        if i%5 == 0
             # rewards1[i], rewards2[i] = test(s -> actor(s), env, FR_price_test, grid_price_test, signal_test, test_horizon)
             save("reward.jld", "epoch", i)
             BSON.@save "actor_model_rl.bson" actor
             BSON.@save "critic_model_rl.bson" critic
        end  ## define the test function
    end
#    println("retire epoch: ", retire_epoch)
    return rewards1, rewards2, actor, critic
end

env = FR_battery{Float64}(state_dim, inner_state_dim)

aDim = length(env.actionspace)
oDim = length(env.observationspace)

# actor = Chain(
#     Dense(oDim, 8, tanh),
#     Dense(8,8,tanh),
#     Dense(8, aDim, tanh)
# )
#
# actor = fmap(f64, actor)
# critic = Chain(
#     Dense(oDim+aDim, 10, relu),
#     Dense(10, 5, relu),
#     Dense(5, 1)
# )
# critic = fmap(f64, critic)
BSON.@load "actor_model.bson" actor
BSON.@load "critic_model.bson" critic

start_time = time()
r1, r2, actor, critic = run!(actor, ADAM(), critic, ADAM(), env)

println("training time is :", time() - start_time)

save("reward.jld", "rewards1", r1, "rewards2", r2, "training_time", time()-start_time)

BSON.@save "actor_model_rl.bson" actor
BSON.@save "critic_model_rl.bson" critic

# using Plots
# plt = Plots.plot(r)
# Plots.savefig("ddpg.pdf")
#
# using Plots
