"""
Code inspired by OpenAI gym implementation of CartPole environment. https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

using MDPs
using Random
using UnPack
import MDPs: action_space, state_space, action_meaning, action_meanings, state, action, reward, reset!, step!, in_absorbing_state

export CartPoleEnv

"""
    ### Description
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
        in the left and right direction on the cart.
    ### Action Space
    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
        of the fixed force the cart is pushed with.
    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |
    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
        the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it
    ### Observation Space
    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
        if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
        if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)
    ### Rewards
    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted.
    ### Starting State
    All observations are assigned a uniformly random value in `(-0.05, 0.05)`
    ### Episode End
    The episode ends if any one of the following occurs:
    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 200
"""
mutable struct CartPoleEnv{T <: AbstractFloat} <: AbstractMDP{Vector{T}, Int}
    const 𝑔::T
    const 𝑚₁::Float64
    const 𝑚ₚ::Float64
    const 𝑙::Float64
    const 𝐹::Float64
    const 𝑑𝑡::Float64
    const θ_threshold::Float64
    const 𝑥_threshold::Float64

    const 𝕊::VectorSpace{T}
    const 𝔸::IntegerSpace

    state::Vector{T}  # position 𝑥, velocity 𝑣, pole_angle θ, pole_angualr_velocity ω
    action::Int
    reward::Float64


    function CartPoleEnv{T}(; gravity::Real=9.8, mass_cart::Real=1.0, mass_pole::Real=0.1, length_pole::Real=0.5, force_magnitude::Real=10.0, dt::Real=0.02, theta_threshold::Real=π/15.0, x_threshold::Real=2.4) where T <: AbstractFloat
        state_upper_bounds = T[2 * x_threshold, Inf, 2 * theta_threshold, Inf]
        state_space = VectorSpace{T}(-state_upper_bounds, state_upper_bounds)
        new{T}(gravity, mass_cart, mass_pole, length_pole, force_magnitude, dt, theta_threshold, x_threshold, state_space, IntegerSpace(2), zeros(T, 4), 1, 0.0)
    end

end

@inline state_space(cp::CartPoleEnv) = cp.𝕊
@inline action_space(cp::CartPoleEnv) = cp.𝔸
const CARTPOLE_ACTION_MEANINGS = ["left", "right"]
@inline action_meaning(::CartPoleEnv, a::Int) = CARTPOLE_ACTION_MEANINGS[a]

function reset!(cp::CartPoleEnv; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    cp.state .= rand(rng, 4) * 0.1 .- 0.05
    @assert cp.state ∈ state_space(cp)
    cp.reward = 0
    cp.action = 1
    nothing
end

function step!(cp::CartPoleEnv, a::Int; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    @assert a ∈ action_space(cp)
    cp.action = a
    if in_absorbing_state(cp)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        cp.reward = 0.0
    else
        @unpack 𝑔, 𝑚₁, 𝑚ₚ, 𝑙, 𝐹, 𝑑𝑡 = cp
        𝑚 = 𝑚₁ + 𝑚ₚ
        𝑥, 𝑣, θ, ω = cp.state
        𝐹 = a == 2 ? 𝐹 : -𝐹
        cosθ = cos(θ)
        sinθ = sin(θ)
        temp = (𝐹 + 𝑚ₚ* 𝑙 * ω^2 * sinθ) / 𝑚
        α = (𝑔 * sinθ - cosθ * temp) / (𝑙 * (4.0 / 3.0 - 𝑚ₚ * cosθ^2 / 𝑚))
        𝑎 = temp - 𝑚ₚ * 𝑙 * α * cosθ / 𝑚

        𝑥 += 𝑣 * 𝑑𝑡
        𝑣 += 𝑎 * 𝑑𝑡
        θ += ω * 𝑑𝑡
        ω += α * 𝑑𝑡

        cp.state.= (𝑥, 𝑣, θ, ω)
        @assert cp.state ∈ state_space(cp)

        cp.reward = 1.0
    end
    nothing
end

function in_absorbing_state(cp::CartPoleEnv)::Bool
    𝑥, _, θ, _ = cp.state
    return abs(𝑥) > cp.𝑥_threshold || abs(θ) > cp.θ_threshold 
end