"""
    ExponentialLikelihood(l::AbstractLink=ExpLink())

Exponential likelihood with scale given by `l(f)`.

```math
    p(y|f) = Exponential(y | l(f))
```
"""
struct ExponentialLikelihood{Tl<:AbstractLink}
    invlink::Tl
end

ExponentialLikelihood() = ExponentialLikelihood(ExpLink())

(l::ExponentialLikelihood)(f::Real) = Exponential(l.invlink(f))

function (l::ExponentialLikelihood)(fs::AbstractVector{<:Real})
    return Product(Exponential.(l.invlink.(fs)))
end
