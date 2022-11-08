"""
Code inspired by OpenAI gym implementation of MountainCar environment. https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
"""

using MDPs
using Random
using UnPack
import MDPs: action_space, state_space, action_meaning, action_meanings, horizon, discount_factor, state, action, reward, reset!, step!, in_absorbing_state

export MountainCarEnv

"""
    ### Description
    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with discrete actions.
    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)
    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```
    ### Observation Space
    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:
    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car                  | -Inf | Inf | position (m) |
    ### Action Space
    There are 3 discrete deterministic actions:
    | Num | Observation             | Value | Unit         |
    |-----|-------------------------|-------|--------------|
    | 0   | Accelerate to the left  | Inf   | position (m) |
    | 1   | Don't accelerate        | Inf   | position (m) |
    | 2   | Accelerate to the right | Inf   | position (m) |
    ### Transition Dynamics:
    Given an action, the mountain car follows the following transition dynamics:
    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*
    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*
    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
    velocity is clipped to the range `[-0.07, 0.07]`.
    ### Reward:
    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep.
    ### Starting State
    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.
    ### Episode End
    The episode ends if either of the following happens:
    1. Termination: The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. Truncation: The length of the episode is 200.
"""
mutable struct MountainCarEnv{T <: AbstractFloat} <: AbstractMDP{Vector{T}, Int}
    const 𝑔::Float64
    const 𝑥_range::Tuple{Float64, Float64}
    const speed_max::Float64
    const 𝑥_goal::Float64
    const 𝑣_goal::Float64
    const 𝐹::Float64

    const horizon::Int
    const 𝕊::VectorSpace{T}
    const 𝔸::IntegerSpace

    state::Vector{T}  # position 𝑥, velocity 𝑣
    action::Int
    reward::Float64

    function MountainCarEnv{T}(; gravity::Real=0.0025, position_range::Tuple{Real, Real}=(-1.2, 0.6), max_speed::Real=0.07, goal_position::Real=0.5, goal_velocity::Real=0, force_magnitude=0.001, horizon::Integer=200) where {T<:AbstractFloat}
        state_lower_bounds = T[position_range[1], -max_speed]
        state_upper_bounds = T[position_range[2], max_speed] 
        state_space = VectorSpace{T}(state_lower_bounds, state_upper_bounds)
        new{T}(gravity, position_range, max_speed, goal_position, goal_velocity, force_magnitude, horizon, state_space, IntegerSpace(3), zeros(T, 2), 1, 0.0)
    end
end

@inline state_space(mc::MountainCarEnv) = mc.𝕊
@inline action_space(mc::MountainCarEnv) = mc.𝔸
const MOUNTAIN_CAR_ACTION_MEANINGS = ["left", "zero", "right"]
@inline action_meaning(::MountainCarEnv, a::Int) = MOUNTAIN_CAR_ACTION_MEANINGS[a]
@inline horizon(mc::MountainCarEnv) = mc.horizon
@inline discount_factor(::MountainCarEnv) = 0.99


function reset!(mc::MountainCarEnv; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    mc.state[1] = rand(rng) * 0.2 - 0.6
    mc.state[2] = 0
    @assert mc.state ∈ state_space(mc)
    nothing
end

function step!(mc::MountainCarEnv, a::Int; rng::AbstractRNG=Random.GLOBAL_RNG)::Nothing
    @assert a ∈ action_space(mc)
    mc.action = a
    if in_absorbing_state(mc)
        @warn "The environment is in an absorbing state. This `step!` will not do anything. Please call `reset!`."
        mc.reward = 0.0
    else
        @unpack 𝑔, 𝑥_range, speed_max, 𝑥_goal, 𝑣_goal, 𝐹 = mc
        𝑥, 𝑣 = mc.state
        𝑣 += (a - 2) * 𝐹 - 𝑔 * cos(3 * 𝑥)
        𝑣 = clamp(𝑣, -speed_max, speed_max)
        𝑥 += 𝑣
        𝑥 = clamp(𝑥, 𝑥_range[1], 𝑥_range[2])
        if 𝑥 == 𝑥_range[1] && 𝑣 < 0
            𝑣 = 0
        end

        mc.state .= (𝑥, 𝑣)
        @assert mc.state ∈ state_space(mc)

        mc.reward = -1.0
    end
    nothing
end

function in_absorbing_state(mc::MountainCarEnv)::Bool
    𝑥, 𝑣 = mc.state
    return 𝑥 >= mc.𝑥_goal && 𝑣 >= mc.𝑣_goal
end