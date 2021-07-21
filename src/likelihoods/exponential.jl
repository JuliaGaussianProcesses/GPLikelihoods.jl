"""
    ExponentialLikelihood()

Exponential likelihood with scale given by `exp(f)`.

```math
    p(y|f) = Exponential(y | exp(f))
```
"""
struct ExponentialLikelihood end

(l::ExponentialLikelihood)(f::Real) = Exponential(exp(f))

(l::ExponentialLikelihood)(fs::AbstractVector{<:Real}) = Product(Exponential.(exp.(fs)))
