abstract type NBParam end

"""
    NegativeBinomialLikelihood(param::NBParam, l::Function/Link)

There are many possible parametrizations for the Negative Binomial likelihood.
The `NegativeBinomialLikelihood` has a special structure, the first type `NBParam`
defines what parametrization is used, and contains the related parameters.
Each `NBParam` has its own documentation:
- [`NBParamI`](@ref): This is the definition used by Distributions.jl.
- [`NBParamII`](@ref): This is the definition used by Wikipedia.
- [`NBParamIII`](@ref): This is the definition based on the mean.

To create a new parametrization, you need to;
- Create a new type `struct MyNBParam{T} <: NBParam; myparams::T; end`
- Dispatch `(l::NegativeBinomialLikelihood{<:MyNBParam})(f::Real)`, which return a [`NegativeBinomial`](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial) from `Distributions.jl`.
`NegativeBinomial` follows the parametrization of [`NBParamI`](@ref), i.e. the first argument is the number of successes
and the second argument is the probability of success.

## Examples

```julia-repl
julia> lik = NegativeBinomialLikelihood(NBParamI(10), logistic)
NegativeBinomialLikelihood{NBParamI{Int64}, LogisticLink}(NBParamI{Int64}(10), LogisticLink(LogExpFunctions.logistic))
julia> lik(2.0)
Distributions.NegativeBinomial{Float64}(r=10.0, p=0.8807970779778824)

julia> lik = NegativeBinomialLikelihood(NBParamIII(10), exp)
NegativeBinomialLikelihood{NBParamIII{Int64}, ExpLink}(NBParamIII{Int64}(10), ExpLink(exp))
julia> lik(2.0)
Distributions.NegativeBinomial{Float64}(r=10.0, p=0.4249256576603398)
```
"""
struct NegativeBinomialLikelihood{Tp<:NBParam,Tl<:AbstractLink} <: AbstractLikelihood
    params::Tp # Likelihood parametrization (and parameters)
    invlink::Tl
    function NegativeBinomialLikelihood(params::Tparam, invlink) where {Tparam}
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
    NBParamI(successes)

Negative Binomial parametrization with `successes` the number of successes and
`l(f)` the probability of `success`.
This corresponds to the definition used by [Distributions.jl](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial).

```math
  p(k|successes, f) = \\frac{\\Gamma(k+successes)}{k! \\Gamma(successes)} l(f)^successes (1 - l(f))^k
```
"""
struct NBParamI{T} <: NBParam
    successes::T
end

function (l::NegativeBinomialLikelihood{<:NBParamI})(f::Real)
    return NegativeBinomial(l.params.successes, l.invlink(f))
end

"""
    NBParamII(failures)

Negative Binomial parametrization with `failures` the number of failures and
`l(f)` the probability of `success`.
This corresponds to the definition used by [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).

```math
  p(k|failures, f) = \\frac{\\Gamma(k+failures)}{k! \\Gamma(failure)} l(f)^k (1 - l(f))^{failures}
```
"""
struct NBParamII{T} <: NBParam
    failures::T
end

function (l::NegativeBinomialLikelihood{<:NBParamII})(f::Real)
    return NegativeBinomial(l.params.failures, 1 - l.invlink(f))
end

"""
    NBParamIII(successes)

Negative Binomial parametrization with mean `μ=l(f)` and number of successes `successes`.
See the definition given in the [Wikipedia article](https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations)
"""
struct NBParamIII{T} <: NBParam
    successes::T
end

function (l::NegativeBinomialLikelihood{<:NBParamIII})(f::Real)
    μ = l.invlink(f)
    r = l.params.successes
    return NegativeBinomial(r, μ / (μ + r))
end
