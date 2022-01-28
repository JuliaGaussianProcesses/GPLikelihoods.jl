"""
    NegBinomialLikelihood(r::Real=1.0, l=logistic)

Negative binomial likelihood with number of successes `r`.

```math
    p(k|f) = \\operatorname{NB}(k | r, l(f))
```
On calling, this would return a negative binomial distribution with `r` successes and probability of success equal to `l(f)`.
Note the distinction of the role of parameter `r` to the Wikipedia definition.
"""
struct NegBinomialLikelihood{T<:Real,Tl<:AbstractLink} <: AbstractLikelihood
    r::T    # number of failures parameter
    invlink::Tl
end

NegBinomialLikelihood(l) = NegBinomialLikelihood(1.0, l)
NegBinomialLikelihood(r::Real=1.0, l=logistic) = NegBinomialLikelihood(r, Link(l))

@functor NegBinomialLikelihood

(l::NegBinomialLikelihood)(f::Real) = NegativeBinomial(l.r, l.invlink(f))

function (l::NegBinomialLikelihood)(fs::AbstractVector{<:Real})
    return Product(NegativeBinomial.(l.r, l.invlink.(fs)))
end
