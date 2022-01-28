"""
    NegativeBinomialLikelihood(r::Real=1.0, l=logistic)

Negative binomial likelihood with number of successes `r`.

```math
    p(k|f) = \\frac{\\Gamma(k+r)}{k! \\Gamma(r)} p^r (1 - p)^k
```
On calling, this would return a negative binomial distribution with `r` successes and probability of success equal to `l(f)`.
Note the distinction of the role of parameter `r` to the Wikipedia definition.
"""
struct NegativeBinomialLikelihood{T<:Real,Tl<:AbstractLink} <: AbstractLikelihood
    r::T    # number of failures parameter
    invlink::Tl
end

NegativeBinomialLikelihood(l=logistic; r::Real=1) = NegativeBinomialLikelihood(r, link(l))

@functor NegativeBinomialLikelihood

(l::NegativeBinomialLikelihood)(f::Real) = NegativeBinomial(l.r, l.invlink(f))

function (l::NegativeBinomialLikelihood)(fs::AbstractVector{<:Real})
    return Product(NegativeBinomial.(l.r, l.invlink.(fs)))
end
