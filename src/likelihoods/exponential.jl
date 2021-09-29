"""
    ExponentialLikelihood(l::AbstractLink=ExpLink())

Exponential likelihood with scale given by `l(f)`.

```math
    p(y|f) = \\operatorname{Exponential}(y | l(f))
```
"""
struct ExponentialLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

ExponentialLikelihood() = ExponentialLikelihood(ExpLink())

(l::ExponentialLikelihood)(f::Real) = Exponential(l.invlink(f))

function (l::ExponentialLikelihood)(fs::AbstractVector{<:Real})
    return Product(Exponential.(l.invlink.(fs)))
end
