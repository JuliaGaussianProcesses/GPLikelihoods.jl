"""
    GammaLikelihood(α::Real=1.0, invlink::AbstractLink=ExpLink())

Gamma likelihood with fixed shape `α`.

```math
    p(y|f) = Gamma(y | α, l(f))
```
On calling, this would return a Gamma distribution with shape `α` and scale `invlink(f)`.
"""
struct GammaLikelihood{Tl<:AbstractLink,T<:Real}
    α::T    # shape parameter
    invlink::Tl
end

GammaLikelihood() = GammaLikelihood(1.0)

GammaLikelihood(α::Real) = GammaLikelihood(α, ExpLink())

@functor GammaLikelihood

(l::GammaLikelihood)(f::Real) = Gamma(l.α, l.invlink(f))

(l::GammaLikelihood)(fs::AbstractVector{<:Real}) = Product(Gamma.(l.α, l.invlink.(fs)))
