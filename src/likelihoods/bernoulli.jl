"""
    BernoulliLikelihood(l::AbstractLink=LogisticLink())

Bernoulli likelihood is to be used if we assume that the 
uncertainity associated with the data follows a Bernoulli distribution.
The link `l` needs to transform the input `f` to the domain [0, 1]

```math
    p(y|f) = \\operatorname{Bernoulli}(y | l(f))
```
On calling, this would return a Bernoulli distribution with `l(f)` probability of `true`.
"""
struct BernoulliLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

BernoulliLikelihood() = BernoulliLikelihood(LogisticLink())

(l::BernoulliLikelihood)(f::Real) = Bernoulli(l.invlink(f))

(l::BernoulliLikelihood)(fs::AbstractVector{<:Real}) = Product(Bernoulli.(l.invlink.(fs)))
