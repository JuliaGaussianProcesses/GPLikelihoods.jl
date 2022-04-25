"""
    NegativeBinomialLikelihood

Abstract base type for the Negative Binomial likelihood. There are multiple
parametrizations; you must choose one of those concrete parameterizations.

You can find all available parametrizations using
```julia
subtypes(NegativeBinomialLikelihood)
```
"""
abstract type NegativeBinomialLikelihood end

"""
## `NBParam` predefined types

### `NBParam` with `p = link(f)` the probability of success
- [`NBParamSuccess`](@ref): This is the definition used in [`Distributions.jl`](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial).
- [`NBParamFailure`](@ref): This is the definition used in [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).


### `NBParam` with `mean = link(f)`
- [`NBParamI`](@ref): Mean is linked to `f` and variance is given by `μ(1 + α)`
- [`NBParamII`](@ref): Mean is linked to `f` and variance is given by `μ(1 + αμ)`
- [`NBParamPower`](@ref): Mean is linked to `f` and variance is given by `μ + αμ^ρ`


To create a new parametrization, you need to;
- Create a new type `struct MyNBParam{T} <: NBParam; myparams::T; end`
- Dispatch `(l::NegativeBinomialLikelihood{<:MyNBParam})(f::Real)`, which return a [`NegativeBinomial`](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial) from `Distributions.jl`.
`NegativeBinomial` follows the parametrization of [`NBParamI`](@ref), i.e. the first argument is the number of successes
and the second argument is the probability of success.

## Examples

```julia-repl
julia> NegativeBinomialLikelihood(NBParamSuccess(10), logistic)(2.0)
NegativeBinomial{Float64}(r=10.0, p=0.8807970779778824)
julia> NegativeBinomialLikelihood(NBParamFailure(10), logistic)(2.0)
NegativeBinomial{Float64}(r=10.0, p=0.11920292202211757)

julia> d = NegativeBinomialLikelihood(NBParamI(3.0), exp)(2.0)
NegativeBinomial{Float64}(r=2.4630186996435506, p=0.25)
julia> mean(d) ≈ exp(2.0)
true
julia> var(d) ≈ exp(2.0) * (1 + 3.0)
true
```

[^1] Hilbe, Joseph M. Negative binomial regression. Cambridge University Press, 2011.
"""

(l::NegativeBinomialLikelihood)(fs::AbstractVector{<:Real}) = Product(map(l, fs))

"""
    NBParamSuccess(successes)

Negative Binomial parametrization with `successes` the number of successes and
`l(f)` the probability of `success`.
This corresponds to the definition used by [Distributions.jl](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial).

```math
  p(k|successes, f) = \\frac{\\Gamma(k+successes)}{k! \\Gamma(successes)} l(f)^successes (1 - l(f))^k
```
"""
struct NBParamSuccess{T,Tl} <: NegativeBinomialLikelihood
    successes::T
    invlink::Tl
end

function (l::NBParamSuccess)(f::Real)
    return NegativeBinomial(l.successes, l.invlink(f))
end

"""
    NBParamFailure(failures)

Negative Binomial parametrization with `failures` the number of failures and
`l(f)` the probability of `success`.
This corresponds to the definition used by [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).

```math
  p(k|failures, f) = \\frac{\\Gamma(k+failures)}{k! \\Gamma(failure)} l(f)^k (1 - l(f))^{failures}
```
"""
struct NBParamFailure{T,Tl} <: NegativeBinomialLikelihood
    failures::T
    invlink::Tl
end

function (l::NBParamFailure)(f::Real)
    return NegativeBinomial(l.failures, 1 - l.invlink(f))
end

# Helper function to convert mean and variance to p and r
_nb_mean_var_to_r_p(μ::Real, v::Real) = abs2(μ) / (v - μ), μ / v

"""
    NBParamI(α)

Negative Binomial parametrization with mean `μ=l(f)` and variance `v=μ(1 + α)`.
"""
struct NBParamI{T,Tl} <: NegativeBinomialLikelihood
    α::T
    invlink::Tl
end

function (l::NBParamI)(f::Real)
    μ = l.invlink(f)
    v = μ * (1 + l.α)
    return NegativeBinomial(_nb_mean_var_to_r_p(μ, v)...)
end

"""
    NBParamII(α)

Negative Binomial parametrization with mean `μ=l(f)` and variance `v=μ(1 + αμ)`.
"""
struct NBParamII{T,Tl} <: NegativeBinomialLikelihood
    α::T
    invlink::Tl
end

function (l::NBParamII)(f::Real)
    μ = l.invlink(f)
    v = μ * (1 + l.α * μ)
    return NegativeBinomial(_nb_mean_var_to_r_p(μ, v)...)
end

"""
    NBParamPower(α, ρ)

Negative Binomial parametrization with mean `μ=l(f)` and variance `v=μ(1 + αμ^ρ)`.
"""
struct NBParamPower{Tα,Tρ,Tl} <: NegativeBinomialLikelihood
    α::Tα
    ρ::Tρ
    invlink::Tl
end

function (l::NBParamPower)(f::Real)
    μ = l.invlink(f)
    v = μ * (1 + l.α * μ^l.ρ)
    return NegativeBinomial(_nb_mean_var_to_r_p(μ, v)...)
end
