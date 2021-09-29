"""
    PoissonLikelihood(l::AbstractLink=ExpLink())

Poisson likelihood with rate defined as `l(f)`.

```math
    p(y|f) = \\mathrm{Poisson}(y | θ=l(f))
```

This is to be used if  we assume that the uncertainity associated
with the data follows a Poisson distribution.
"""
struct PoissonLikelihood{L<:AbstractLink} <: AbstractLikelihood
    invlink::L
end

PoissonLikelihood() = PoissonLikelihood(ExpLink())

(l::PoissonLikelihood)(f::Real) = Poisson(l.invlink(f))

(l::PoissonLikelihood)(fs::AbstractVector{<:Real}) = Product(Poisson.(l.invlink.(fs)))
