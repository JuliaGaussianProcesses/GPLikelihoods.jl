"""
    NegativeBinomialLikelihood(l=logistic; successes::Real=1)

Negative binomial likelihood with number of successes `successes`.

```math
    p(k|successes, f) = \\frac{\\Gamma(k+successes)}{k! \\Gamma(successes)} l(f)^successes (1 - l(f))^k
```
On calling, this returns a negative binomial distribution with `successes` successes and 
probability of success equal to `l(f)`.

!!! warning "Parameterization" 
    The parameter `successes` is different from the parameter `r` in the 
    [Wikipedia definition](http://en.wikipedia.org/wiki/Negative_binomial_distribution), 
    which denotes the number of failures.
    This parametrization is used in order to stay consistent with the parametrization in 
    [Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.NegativeBinomial).
    To use the Wikipedia definition, set `successes` as the number of "failures" and
    change the probability of success from `l(f)` to `1 - l(f)`.
    Note that with symmetric functions like the [`LogisticLink`](@ref), this corresponds to
    using `l(-f)`.
"""
struct NegativeBinomialLikelihood{Tl<:AbstractLink,T<:Real} <: AbstractLikelihood
    successes::T    # number of successes parameter
    invlink::Tl
end

function NegativeBinomialLikelihood(l=logistic; successes::Real=1)
    return NegativeBinomialLikelihood(successes, link(l))
end

@functor NegativeBinomialLikelihood

(l::NegativeBinomialLikelihood)(f::Real) = NegativeBinomial(l.successes, l.invlink(f))

(l::NegativeBinomialLikelihood)(fs::AbstractVector{<:Real}) = Product(map(l, fs))
