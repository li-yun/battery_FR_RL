module RLEnvironments
    using Distributions: Uniform
    import Distributions.sample

    abstract type Environment{S,A,R<:Real} end
    abstract type DiscEnv{S,A,R} <: Environment{S,A,R} end
    abstract type ContEnv{S,A,R} <: Environment{S,A,R} end
    abstract type DiffEnv{S,A,R} <: Environment{S,A,R} end
    function step! end
    function reset! end
    function set! end
#    export Environment, DiscEnv, ContEnv, DiffEnv, step!, reset!, ActionSpace, ObservationSpace,step!,reset!
    #DiscEnv: Vector{A}, ContEnv: Vector{Tuple{A, A}}
    # struct ActionSpace{A}
    #     actions::Vector{A}
    # end
    struct ActionSpace{A}
        actions::Vector{Tuple{A,A}}
    end

    # ActionSpace(n::A) where A <: Real = ActionSpace(collect(one(A):n))

    Base.length(a::ActionSpace) = length(a.actions)
    Base.eltype(::Type{ActionSpace{A}}) where A = A
#    Base.eltype(::Type{ActionSpace{Tuple{A, A}}}) where A = A

    # sample(a::ActionSpace) = rand(a.actions)
    function sample(a::ActionSpace{T}) where T <: AbstractFloat
        return map(x -> T(rand(Uniform(x[1], x[2]))), a.actions)
    end

    #n = numstates // n = length(state)
    struct ObservationSpace{S}
        n::Int64
    end

    Base.length(o::ObservationSpace) = o.n
    Base.eltype(::Type{ObservationSpace{S}}) where S = S

    # test function get a/multiple period of FR_price, grid_price and signal as additional input
    # include("bandit.jl")
    # include("simplerooms.jl")
    # include("mountaincar.jl")
    # include("cartpole.jl")
    # include("pendulum.jl")
    include("battery.jl")
    export Environment, DiscEnv, ContEnv, DiffEnv, step!, reset!, ActionSpace, ObservationSpace, set!
    export FR_battery
end
