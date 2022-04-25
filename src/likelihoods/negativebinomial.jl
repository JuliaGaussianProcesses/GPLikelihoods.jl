abstract type NBParam end
default_invlink(::NBParam) = error("Not implemented")

abstract type NBParamProb <: NBParam end
default_invlink(::NBParamProb) = logistic
abstract type NBParamMean <: NBParam end
default_invlink(::NBParamMean) = exp

"""
    NegativeBinomialLikelihood(param::NBParam, l::Function/Link)

There are many possible parametrizations for the Negative Binomial likelihood.
We follow the convention laid out in p.137 of [^1] and some common parametrizations.
The `NegativeBinomialLikelihood` has a special structure, the first type `NBParam`
defines what parametrization is used, and contains the related parameters.
`NBParam` itself has two subtypes:
- `NBParamProb` for parametrizations where `f->p`, the probability of success
- `NBParamMean` for parametrizations where `f->μ`, the mean

## `NBParam` predefined types

### `NBParamProb` types with `p = link(f)` the probability of success
- [`NBParamSuccess`](@ref): This is the definition used in [`Distributions.jl`](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial).
- [`NBParamFailure`](@ref): This is the definition used in [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).


### `NBParamMean` types with `mean = link(f)`
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
struct NegativeBinomialLikelihood{Tp<:NBParam,Tl<:AbstractLink} <: AbstractLikelihood
    params::Tp # Likelihood parametrization (and parameters)
    invlink::Tl
    function NegativeBinomialLikelihood(params::Tparam, invlink=default_invlink(params)) where {Tparam<:NBParam}
        invlink = link(invlink)
        return new{Tparam,typeof(invlink)}(params, invlink)
    end
end

@deprecate NegativeBinomialLikelihood(l=logistic; successes=1) NegativeBinomialLikelihood(
    NBParamI(successes), l
)

function (l::NegativeBinomialLikelihood)(::Real)
    return error(
        "not implemented for type $(typeof(l)). See `NegativeBinomialLikelihood` docs"
    )
end

@functor NegativeBinomialLikelihood

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
struct NBParamSuccess{T} <: NBParamProb
    successes::T
end

function (l::NegativeBinomialLikelihood{<:NBParamSuccess})(f::Real)
    return NegativeBinomial(l.params.successes, l.invlink(f))
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
struct NBParamFailure{T} <: NBParamProb
    failures::T
end

function (l::NegativeBinomialLikelihood{<:NBParamFailure})(f::Real)
    return NegativeBinomial(l.params.failures, 1 - l.invlink(f))
end

# Helper function to convert mean and variance to p and r
_nb_mean_var_to_r_p(μ::Real, v::Real) = abs2(μ) / (v - μ), μ / v

"""
    NBParamI(α)

Negative Binomial parametrization with mean `μ=l(f)` and variance `v=μ(1 + α)`.
"""
struct NBParamI{T} <: NBParamMean
    α::T
end

function (l::NegativeBinomialLikelihood{<:NBParamI})(f::Real)
    μ = l.invlink(f)
    v = μ * (1 + l.params.α)
    return NegativeBinomial(_nb_mean_var_to_r_p(μ, v)...)
end

"""
    NBParamII(α)

Negative Binomial parametrization with mean `μ=l(f)` and variance `v=μ(1 + αμ)`.
"""
struct NBParamII{T} <: NBParamMean
    α::T
end

function (l::NegativeBinomialLikelihood{<:NBParamII})(f::Real)
    μ = l.invlink(f)
    v = μ * (1 + l.params.α * μ)
    return NegativeBinomial(_nb_mean_var_to_r_p(μ, v)...)
end

"""
    NBParamPower(α, ρ)

Negative Binomial parametrization with mean `μ=l(f)` and variance `v=μ(1 + αμ^ρ)`.
"""
struct NBParamPower{Tα,Tρ} <: NBParamMean
    α::Tα
    ρ::Tρ
end

function (l::NegativeBinomialLikelihood{<:NBParamPower})(f::Real)
    μ = l.invlink(f)
    v = μ * (1 + l.params.α * μ^l.params.ρ)
    return NegativeBinomial(_nb_mean_var_to_r_p(μ, v)...)
end
