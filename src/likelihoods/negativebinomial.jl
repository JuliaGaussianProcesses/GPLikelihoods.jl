abstract type NBParam end

"""
    NegativeBinomialLikelihood{NBParam}(l=logistic/exp; kwargs...)

There are many possible parametrizations for the Negative Binomial likelihood.
The `NegativeBinomialLikelihood` has a special structure, the first type `NBParam`
defines what parametrization is used.
Each `NBParam` has its own documentation:
- [`NBParamI`](@ref): This is the definition used by Distributions.jl.
- [`NBParamII`](@ref): This is the definition used by Wikipedia.
- [`NBParamIII`](@ref): This is the definition based on the mean.

To create a new parametrization, you need to;
- Create a new typee from `struct MyNBParam <: NBParam end`
- Dispatch [`to_dist_params`](@ref): `to_dist_params(l::NegativeBinomialLikelihood{MyNBParam}, f::Real)`
to return the parameters for the [`NegativeBinomial`](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial) constructor
- Write a constructor for `NegativeBinomialLikelihood{MyNBParam}`
"""
struct NegativeBinomialLikelihood{NBParam,Tl<:AbstractLink,T} <: AbstractLikelihood
    params::T # Likelihood parameters (depends of NBParam)
    invlink::Tl
    function NegativeBinomialLikelihood{Tparam}(params, invlink) where {Tparam}
        invlink = link(invlink)
        return new{Tparam,typeof(invlink),typeof(params)}(params, invlink)
    end
end

@deprecate NegativeBinomialLikelihood(l=logistic; successes=1) NegativeBinomialLikelihood{
    NBParamI
}(
    l; successes
)

@functor NegativeBinomialLikelihood

function (l::NegativeBinomialLikelihood)(f::Real)
    return NegativeBinomial(to_dist_params(l, f)...)
end

(l::NegativeBinomialLikelihood)(fs::AbstractVector{<:Real}) = Product(map(l, fs))

"""
    to_dist_params(l::NegativeBinomialLikelihood{NBParam}, f::Real)->Tuple{Real,Real}

Take parameters from a given parametrization `NBParam` to the parametrization
used by `Distributions.jl`
"""
to_dist_params

"""
    NBParamI

Negative Binomial parametrization with `successes` the number of successes and
`l(f)` the probability of `success`.
This corresponds to the definition used by [Distributions.jl](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.NegativeBinomial).

```math
  p(k|successes, f) = \\frac{\\Gamma(k+successes)}{k! \\Gamma(successes)} l(f)^successes (1 - l(f))^k
```
"""
struct NBParamI <: NBParam end

function to_dist_params(l::NegativeBinomialLikelihood{NBParamI}, f::Real)
    return l.params.successes, l.invlink(f)
end

function NegativeBinomialLikelihood{NBParamI}(l=logistic; successes::Real=1)
    return NegativeBinomialLikelihood{NBParamI}((; successes), l)
end

"""
    NBParamII

Negative Binomial parametrization with `failures` the number of failures and
`l(f)` the probability of `success`.
This corresponds to the definition used by [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution).

```math
  p(k|failures, f) = \\frac{\\Gamma(k+failures)}{k! \\Gamma(failure)} l(f)^k (1 - l(f))^{failures}
```
"""
struct NBParamII <: NBParam end

function to_dist_params(l::NegativeBinomialLikelihood{NBParamII}, f::Real)
    return l.params.failures, 1 - l.invlink(f)
end

function NegativeBinomialLikelihood{NBParamII}(l=logistic; failures::Real=1)
    return NegativeBinomialLikelihood{NBParamII}((; failures), l)
end

"""
    NBParamIII

Negative Binomial parametrization with mean `μ=l(f)` and number of successes `successes`.
See the definition given in the [Wikipedia article](https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations)
"""
struct NBParamIII <: NBParam end

function to_dist_params(l::NegativeBinomialLikelihood{NBParamIII}, f::Real)
    μ = l.invlink(f)
    r = l.params.successes
    return r, μ / (μ + r)
end

function NegativeBinomialLikelihood{NBParamIII}(l=exp; successes::Real=1)
    return NegativeBinomialLikelihood{NBParamIII}((; successes), l)
end
