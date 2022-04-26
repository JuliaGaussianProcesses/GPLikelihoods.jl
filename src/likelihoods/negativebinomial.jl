abstract type NBParam end

abstract type NBParamProb <: NBParam end
default_invlink(::NBParamProb) = logistic
abstract type NBParamMean <: NBParam end
default_invlink(::NBParamMean) = exp

"""
    NegativeBinomialLikelihood(param::NBParam, invlink::Union{Function,Link})

There are many possible parametrizations for the Negative Binomial likelihood.
We follow the convention laid out in p.137 of [^1] and provide some common parametrizations.
The `NegativeBinomialLikelihood` has a special structure; the first type parameter `NBParam`
defines in what parametrization the latent function is used, and contains the other (scalar) parameters.
`NBParam` itself has two subtypes:
- `NBParamProb` for parametrizations where `f -> p`, the probability of success of a Bernoulli event
- `NBParamMean` for parametrizations where `f -> μ`, the expected number of events

## `NBParam` predefined types

### `NBParamProb` types with `p = invlink(f)` the probability of success
- [`NBParamSuccess`](@ref): This is the definition used in [`Distributions.jl`](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial).
- [`NBParamFailure`](@ref): This is the definition used in [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).


### `NBParamMean` types with `μ = invlink(f)` the mean/expected number of events
- [`NBParamI`](@ref): Mean is linked to `f` and variance is given by `μ(1 + α)`
- [`NBParamII`](@ref): Mean is linked to `f` and variance is given by `μ(1 + α * μ)`
- [`NBParamPower`](@ref): Mean is linked to `f` and variance is given by `μ(1 + α * μ^ρ)`


To create a new parametrization, you need to:
- create a new type `struct MyNBParam{T} <: NBParam; myparams::T; end`;
- dispatch `(l::NegativeBinomialLikelihood{<:MyNBParam})(f::Real)`, which must return a [`NegativeBinomial`](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial) from `Distributions.jl`.
`NegativeBinomial` follows the parametrization of [`NBParamSuccess`](@ref), i.e. the first argument is the number of successes
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
struct NegativeBinomialLikelihood{Tp<:NBParam,Tl<:AbstractLink} <: AbstractLikelihood
    params::Tp # Likelihood parametrization (and parameters)
    invlink::Tl
    function NegativeBinomialLikelihood(
        params::Tparam, invlink=default_invlink(params)
    ) where {Tparam<:NBParam}
        # we convert `invlink` into a `Link` object if it's not the case already
        invlink = link(invlink)
        return new{Tparam,typeof(invlink)}(params, invlink)
    end
end

@deprecate NegativeBinomialLikelihood(l=logistic; successes=1) NegativeBinomialLikelihood(
    NBParamI(successes), l
)

function (l::NegativeBinomialLikelihood)(::Real)
    return error(
        "not implemented for type $(typeof(l)). For your custom type to run you ",
        "need to implement `(l::NegativeBinomialLikelihood{<:MyNBParam})(f::Real)`. ",
        "For a full explanation, see `NegativeBinomialLikelihood` docs",
    )
end

@functor NegativeBinomialLikelihood

(l::NegativeBinomialLikelihood)(fs::AbstractVector{<:Real}) = Product(map(l, fs))

@doc raw"""
    NBParamSuccess(successes)

Negative Binomial parametrization with `successes` the number of successes and
`invlink(f)` the probability of success.
This corresponds to the definition used by [Distributions.jl](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial).

```math
  p(y|\text{successes}, p=invlink(f)) = \frac{\Gamma(y+\text{successes})}{y! \Gamma(\text{successes})} p^\text{successes} (1 - p)^y
```
"""
struct NBParamSuccess{T} <: NBParamProb
    successes::T
end

function (l::NegativeBinomialLikelihood{<:NBParamSuccess})(f::Real)
    return NegativeBinomial(l.params.successes, l.invlink(f))
end

@doc raw"""
    NBParamFailure(failures)

Negative Binomial parametrization with `failures` the number of failures and
`invlink(f)` the probability of success.
This corresponds to the definition used by [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).

```math
  p(y|\text{failures}, p=\text{invlink}(f)) = \frac{\Gamma(y+\text{failures})}{y! \Gamma(\text{failures})} p^y (1 - p)^{\text{failures}}
```
"""
struct NBParamFailure{T} <: NBParamProb
    failures::T
end

function (l::NegativeBinomialLikelihood{<:NBParamFailure})(f::Real)
    return NegativeBinomial(l.params.failures, 1 - l.invlink(f))
end

# Helper function to convert mean and variance to p and r
_nb_mean_excessvar_to_r_p(μ::Real, ev::Real) = μ / ev, 1 / (1 + ev)

"""
    NBParamI(α)

Negative Binomial parametrization with mean `μ=invlink(f)` and variance `v=μ(1 + α)`.
"""
struct NBParamI{T} <: NBParamMean
    α::T
end

function (l::NegativeBinomialLikelihood{<:NBParamI})(f::Real)
    μ = l.invlink(f)
    ev = l.params.α
    return NegativeBinomial(_nb_mean_excessvar_to_r_p(μ, ev)...)
end

"""
    NBParamII(α)

Negative Binomial parametrization with mean `μ=invlink(f)` and variance `v=μ(1 + α * μ)`.
"""
struct NBParamII{T} <: NBParamMean
    α::T
end

function (l::NegativeBinomialLikelihood{<:NBParamII})(f::Real)
    μ = l.invlink(f)
    ev = l.params.α * μ
    return NegativeBinomial(_nb_mean_excessvar_to_r_p(μ, ev)...)
end

"""
    NBParamPower(α, ρ)

Negative Binomial parametrization with mean `μ = invlink(f)` and variance `v = μ(1 + α * μ^ρ)`.
"""
struct NBParamPower{Tα,Tρ} <: NBParamMean
    α::Tα
    ρ::Tρ
end

function (l::NegativeBinomialLikelihood{<:NBParamPower})(f::Real)
    μ = l.invlink(f)
    ev = l.params.α * μ^l.params.ρ
    return NegativeBinomial(_nb_mean_excessvar_to_r_p(μ, ev)...)
end
