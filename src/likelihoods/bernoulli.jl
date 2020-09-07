"""
    BernoulliLikelihood

Bernoulli likelihood is to be used if we assume that the 
uncertainity associated with the data follows a Bernoulli distribution.

```math
    p(y|f) = Bernoulli(y | f)
```
On calling, this would return a Bernoulli distribution with `f` probability of `true`.
"""
struct BernoulliLikelihood end

@functor BernoulliLikelihood

_logistic(logit::Real) = 1/(1 + exp(-logit))

(l::BernoulliLikelihood)(f::Real) = Bernoulli(_logistic(f))

(l::BernoulliLikelihood)(fs::AbstractVector{<:Real}) = Product(Bernoulli.(_logistic.(fs)))
