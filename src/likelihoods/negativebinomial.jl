"""
    NegativeBinomialLikelihood(l=logistic; successes::Real=1)

Negative binomial likelihood with number of successes `successes`.

```math
    p(k|f) = \\frac{\\Gamma(k+successes)}{k! \\Gamma(successes)} l(f)^successes (1 - l(f))^k
```
On calling, this returns a negative binomial distribution with `successes` successes and 
probability of success equal to `l(f)`.

!!! warning "Parameterization" 
    The parameter `successes` is different from the parameter `r` in the 
    [Wikipedia definition](http://en.wikipedia.org/wiki/Negative_binomial_distribution), 
    which denotes the number of failures.
    This parametrization is used in order to keep up with the parametrization in 
    [Distributions.jl](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.NegativeBinomial).

    This change of parameterization is equivalent to changing the number of successes 
    from `p` to `1-p`, which remains unidentifiable when using a link function for which
    `l(-f) = 1 - l(f)` holds, such as `logistic`. Thus, when using such link functions,
    the fact that this implementation uses a non-standard parameterization does not have any 
    downsides.
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

function (l::NegativeBinomialLikelihood)(fs::AbstractVector{<:Real})
    return Product(NegativeBinomial.(l.successes, l.invlink.(fs)))
end
