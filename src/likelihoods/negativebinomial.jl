"""
    NegativeBinomialLikelihood(l=logistic; successes::Real=1)

Negative binomial likelihood with number of successes `successes`.

```math
    p(k|f) = \\frac{\\Gamma(k+successes)}{k! \\Gamma(successes)} l(f)^successes (1 - l(f))^k
```
On calling, this returns a negative binomial distribution with `successes` successes and probability of success equal to `l(f)`.
Note the distinction of the role of parameter `successes` to the parameter `r` in the Wikipedia definition.
"""
struct NegativeBinomialLikelihood{Tl<:AbstractLink,T<:Real} <: AbstractLikelihood
    successes::T    # number of successes parameter
    invlink::Tl
end

NegativeBinomialLikelihood(l=logistic; successes::Real=1) = NegativeBinomialLikelihood(successes, link(l))

@functor NegativeBinomialLikelihood

(l::NegativeBinomialLikelihood)(f::Real) = NegativeBinomial(l.successes, l.invlink(f))

function (l::NegativeBinomialLikelihood)(fs::AbstractVector{<:Real})
    return Product(NegativeBinomial.(l.successes, l.invlink.(fs)))
end
