"""
    GammaLikelihood(α::Real=1.0, l=exp)

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

GammaLikelihood(l) = GammaLikelihood(1.0, l)
GammaLikelihood(α::Real=1.0, l=exp) = GammaLikelihood(α, link(l))

@functor GammaLikelihood

(l::GammaLikelihood)(f::Real) = Gamma(l.α, l.invlink(f))

(l::GammaLikelihood)(fs::AbstractVector{<:Real}) = Product(Gamma.(l.α, l.invlink.(fs)))
