"""
    GammaLikelihood(α)

Gamma likelihood with fixed shape `α`.

```math
    p(y|f) = Gamma(y | α, θ=exp(f))
```
On calling, this would return a gamma distribution with shape `α` and scale `exp(f)`.
"""
struct GammaLikelihood{T<:Real}
    α::T    # shape parameter
end

GammaLikelihood() = GammaLikelihood(1.)

@functor GammaLikelihood

(l::GammaLikelihood)(f::Real) = Gamma(l.α, exp(f))

(l::GammaLikelihood)(fs::AbstractVector{<:Real}) = Product(Gamma.(l.α, exp.(fs)))
