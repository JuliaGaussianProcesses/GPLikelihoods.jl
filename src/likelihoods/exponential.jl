"""
    ExponentialLikelihood(l=exp)

Exponential likelihood with scale given by `l(f)`.

```math
    p(y|f) = \\operatorname{Exponential}(y | l(f))
```
"""
struct ExponentialLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

ExponentialLikelihood(l=exp) = ExponentialLikelihood(link(l))

(l::ExponentialLikelihood)(f::Real) = Exponential(l.invlink(f))

function (l::ExponentialLikelihood)(fs::AbstractVector{<:Real})
    return Product(Exponential.(l.invlink.(fs)))
end
