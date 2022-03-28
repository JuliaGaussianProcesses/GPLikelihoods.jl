"""
    PoissonLikelihood(l=exp)

Poisson likelihood with rate defined as `l(f)`.

```math
    p(y|f) = \\operatorname{Poisson}(y | θ=l(f))
```

This is to be used if  we assume that the uncertainity associated
with the data follows a Poisson distribution.
"""
struct PoissonLikelihood{L<:AbstractLink} <: AbstractLikelihood
    invlink::L
end

PoissonLikelihood(l=exp) = PoissonLikelihood(link(l))

(l::PoissonLikelihood)(f::Real) = Poisson(l.invlink(f))

(l::PoissonLikelihood)(fs::AbstractVector{<:Real}) = Product(map(l, fs))
