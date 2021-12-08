using Statistics
#using PyPlot
using DelimitedFiles
using Distributions
using LinearAlgebra

FR_price = readdlm("FR/FR_Incentive.csv", ',', Float64)
FR_price_orig = vcat(FR_price'...)
FR_price = FR_price_orig.+1e-4
#FR_price = repeat(FR_price; outer=[nyears])

grid_price = readdlm("FR/slow_Price.csv", ',', Float64) # size = 365*24
grid_price_orig = vcat(grid_price'...)

Files = ["FR/01_2017_Dynamic.csv", "FR/02_2017_Dynamic.csv", "FR/03_2017_Dynamic.csv", "FR/04_2017_Dynamic.csv",
            "FR/05_2017_Dynamic.csv", "FR/06_2017_Dynamic.csv", "FR/07_2017_Dynamic.csv", "FR/08_2017_Dynamic.csv",
            "FR/09_2017_Dynamic.csv", "FR/10_2017_Dynamic.csv", "FR/11_2017_Dynamic.csv", "FR/12_2017_Dynamic.csv"]

global signal = []

for i in 1:12
    filename = Files[i]
    originalsignal = readdlm(filename, ',', Float64)[1:(end-1),:]   #positive means the grid sends power to battery (charging) and negative means grid buys power from battery(discharging).
    if i == 1
        global signal = vcat(originalsignal...)
    else
        global signal = [signal; vcat(originalsignal...)]
    end
    if i == 12
       global signal = [signal; originalsignal[end,end]]
    end
end
signal = min.(max.(signal, -1),1)


### calculate the distribution of grid price

grid_price = grid_price[1:364, :]'

#@assert size(grid_price) == (24, 364)

Xn_price = reshape(grid_price, (:, 52)) # size of Xn_price = (168,52)
p, n = size(Xn_price)
Sn = Xn_price * Xn_price' / n
In = Diagonal(ones(p))

m_n = tr(Sn * In) / p
d_n2 = norm(Sn - m_n * In)^2
b_nbar = (1/n^2)*sum(norm(Xn_price[:, i] * Xn_price[:, i]' - Sn)^2 for i = 1:n)
b_n2 = min(b_nbar, d_n2)
a_n2 = d_n2 - b_n2

Sigma_n = (b_n2 / d_n2) * m_n * In + (a_n2 / d_n2) * Sn
mu_n = mean(Xn_price, dims = 2)
mu_n = dropdims(mu_n, dims = 2)
d_grid_price = MvNormal(mu_n, Sigma_n)
#grid_price_sample = rand(d_grid_price, 52)
#grid_price_sample = vcat(grid_price_sample...)
# calculate the distribution of FR price signal

FR_price_mean = mean(log.(FR_price))
FR_price_var = std(log.(FR_price))

d_FR_price = LogNormal(FR_price_mean, FR_price_var)

max_grid_price = maximum(vcat(grid_price...))
min_grid_price = minimum(vcat(grid_price...))
max_FR_price = maximum(FR_price_orig)
min_FR_price = minimum(FR_price_orig)
#FR_price_sample = rand(d_FR_price, 364*24)

# calculate the distribution of FR signal
# global signal_hour = []
# global signal_hour_variance = []
# for i in 1:size(signal)[1]
#     if i%1800==0
#         global signal_hour = vcat(signal_hour, mean(signal[i-1799:i]))
#         global signal_hour_variance = vcat(signal_hour_variance, var(signal[i-1799:i]))
#     end
# end
# mean_signal = mean(signal_hour)
# var_signal = std(signal_hour)
# subplot(2,1,1)
# plot(signal_hour)
# subplot(2,1,2)
# plot(signal_hour_variance)
# PyPlot.display_figs()
# d_signal = Normal(mean_signal, var_signal)
# signal_sample = rand(d_signal, 1800)
signal_index = collect(1:1800:(size(signal)[1]-1800*24*28))
