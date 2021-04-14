"""
    BernoulliLikelihood

Bernoulli likelihood is to be used if we assume that the 
uncertainity associated with the data follows a Bernoulli distribution.

```math
    p(y|f) = Bernoulli(y | f)
```
On calling, this would return a Bernoulli distribution with `f` probability of `true`.
"""
struct BernoulliLikelihood{Tl<:AbstractLink}
    invlink::Tl
end

(l::BernoulliLikelihood)(f::Real) = Bernoulli(l.invlink(f))

(l::BernoulliLikelihood)(fs::AbstractVector{<:Real}) = Product(Bernoulli.(l.invlink.(fs)))
