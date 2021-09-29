"""
    GammaLikelihood(α::Real=1.0, l::AbstractLink=ExpLink())

Gamma likelihood with fixed shape `α`.

```math
    p(y|f) = \\operatorname{Gamma}(y | α, l(f))
```
On calling, this would return a gamma distribution with shape `α` and scale `l(f)`.
"""
struct GammaLikelihood{T<:Real,Tl<:AbstractLink} <: AbstractLikelihood
    α::T    # shape parameter
    invlink::Tl
end

GammaLikelihood() = GammaLikelihood(1.0)

GammaLikelihood(α::Real) = GammaLikelihood(α, ExpLink())

@functor GammaLikelihood

(l::GammaLikelihood)(f::Real) = Gamma(l.α, l.invlink(f))

(l::GammaLikelihood)(fs::AbstractVector{<:Real}) = Product(Gamma.(l.α, l.invlink.(fs)))
