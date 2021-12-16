# Pkg.activate("C:\\Users\\86156\\julia_bat_env_new")
# cd("C:\\Users\\86156\\Desktop\\CC_battery\\nn_train")
using JLD
using Flux
using BSON: @save

filename = "TrainData.jld"
f = load(filename)

X_TRAIN = f["X"]
Y_TRAIN = f["Y"]
Stage_Cost = f["COST"]

# area = 0.3108
# TC = 2.3/area
# Qmax = TC
# P_nominal = TC*3.1
# maxC = 10
#
# # u = [FR_band, grid_band]
# umax = 0.5*[P_nominal*maxC, P_nominal*maxC]
# umin = 0.5*[0, -P_nominal*maxC]


# function Normalize_input(u)
#     return return (u.-   (umin .+ umax) ./ 2      )./((umax-umin)./2)
# end
#
# function Denormalize_input(u)
#     return (umax .+ umin)/2 .+ (umax .- umin)/2 .* u
# end

# output of the nn controller is FR_band, and bug_from_grid


actor = Chain(Dense(6,30,relu),
                     Dense(30,15,relu),
                    Dense(15,2,tanh))

loss1(x, y) = Flux.mae(actor(x), y)   #sum((ml(x) .- y).^2)
# loss2(x, y) = Flux.mae(actor_nn(x), y)
# mae(x, y) = mean(abs.(actor_nn(x).- y))

## train actor network
data_size = size(X_TRAIN)[2]
data = [(X_TRAIN[:,i], Y_TRAIN[:,i]) for i = 1:data_size]
#data = zip(x_train, y_train)
opt = ADAM()
Flux.@epochs 5000 Flux.train!(loss1, Flux.params(actor), data, opt)

@save "actor_model.bson" actor


### train critic network

function data_sample(state, action, cost,batch_size=24000)
    ind = 1:size(state)[2]-1
    s1, a1, r = state[:, ind], action[:, ind], cost[ind]
    s2, a2 = state[:, ind .+ 1], action[:, ind .+ 1]
    return s1, a1, r, s2, a2
end

function softupdate!(target::T, model::T, τ=1f-2) where T
    for f in fieldnames(T)
        softupdate!(getfield(target, f), getfield(model, f), τ)
    end
end

function softupdate!(dst::A, src::A, τ=T(1f-2)) where {T, A<:AbstractArray{T}}
    dst .= τ .* src .+ (one(T) - τ) .* dst
end

oDim, aDim = 6, 2
critic = Chain(
    Dense(oDim+aDim, 30, relu),
    Dense(30, 15, relu),
    Dense(15, 1)
)

critic_nn_tar = deepcopy(critic)

global γ = 0.9
global epoches = 2000
global steps = 200
s1, a1, r, s2, a2 = data_sample(X_TRAIN, Y_TRAIN, Stage_Cost)

for i in 1: epoches
    for j in 1:steps
        Q_tar = r .+ γ * vec(critic_nn_tar(vcat(s2,a2)))
            grads = Flux.gradient(Flux.params(critic)) do
                Q = vec(critic(vcat(s1,a1)))
                return Flux.mae(Q,Q_tar)
            end
            Flux.Optimise.update!(ADAM(), Flux.params(critic), grads)
            softupdate!(critic_nn_tar,critic)
    end
    println("epoch: ", i)
end

@save "critic_model.bson" critic
