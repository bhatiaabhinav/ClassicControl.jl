"""
Code inspired by OpenAI gym implementation of CartPole environment. https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
"""

using MDPs
using Random
using UnPack
import MDPs: action_space, state_space, action_meaning, action_meanings, state, action, reward, reset!, step!, in_absorbing_state

export PendulumEnv

"""
    ### Description
    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.
    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.
    ![Pendulum Coordinate System](./diagrams/pendulum.png)
    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.
    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |
    ### Observation Space
    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.
    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |
    ### Rewards
    The reward function is defined as:
    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*
    where theta is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).
    ### Starting State
    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.
    ### Episode Truncation
    The episode truncates at 200 time steps.
    ### Arguments
    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
    The default value is g = 10.0 .
    ```
    gym.make('Pendulum-v1', g=9.81)
    ```
    ### Version History
    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)
"""
Base.@kwdef mutable struct PendulumEnv{T <: AbstractFloat} <: AbstractMDP{Vector{T}, Int}
    const 𝑔::Float64 = 10.0
    const 𝑣_max::Float64 = 8.0
    const 𝑚::Float64 = 1.0
    const 𝑙::Float64 = 1.0
    const 𝑑𝑡::Float64 = 0.05

    const 𝕊::VectorSpace{T} = VectorSpace{T}(T[-1.0, -1.0, -𝑣_max], T[1.0, 1.0, 𝑣_max])
    const 𝔸::IntegerSpace = IntegerSpace(3)

    θ::Float64 = 0.0
    state::Vector{T} = zeros(T, 3)  # position 𝑥, velocity 𝑣, pole_angle θ, pole_angualr_velocity ω
    action::Int = 1
    reward::Float64 = 0.0
end

@inline state_space(cp::PendulumEnv) = cp.𝕊
@inline action_space(cp::PendulumEnv) = cp.𝔸
const PENDULUM_ACTION_MEANINGS = ["τ=-2", "τ=0", "τ=2"]
@inline action_meaning(::PendulumEnv, a::Int) = PENDULUM_ACTION_MEANINGS[a]

function reset!(cp::PendulumEnv; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    cp.θ = 2π * rand(rng) - 1
    cp.state .= (cos(cp.θ), sin(cp.θ), 2 * rand(rng) - 1)
    @assert cp.state ∈ state_space(cp)
    cp.reward = 0
    cp.action = 1
    nothing
end

function step!(cp::PendulumEnv, a::Int; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    @assert a ∈ action_space(cp)
    cp.action = a
    if in_absorbing_state(cp)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        cp.reward = 0.0
    else
        @unpack 𝑔, 𝑣_max, 𝑚, 𝑙, 𝑑𝑡, θ = cp
        # println(cp.θ)
        cosθ, sinθ, ω = cp.state
        τ::Float64 = (a - 2) * 2  # -2, 0 or 2
        cost = angle_normalize(θ)^2 + 0.1 * ω^2 + 0.001 * (τ^2)

        α = 3(𝑔*sinθ/2𝑙 + τ/(𝑚*𝑙^2)) * 𝑑𝑡
        ω += α * 𝑑𝑡
        ω = clamp(ω, -𝑣_max, 𝑣_max)
        cp.θ += ω * 𝑑𝑡

        cp.state.= (cos(cp.θ), sin(cp.θ), 2 * rand(rng) - 1)
        @assert cp.state ∈ state_space(cp)
        cp.reward = -cost
    end
    nothing
end

in_absorbing_state(cp::PendulumEnv)::Bool = false
angle_normalize(θ::Float64) = Base.mod((θ + π), (2π)) - π