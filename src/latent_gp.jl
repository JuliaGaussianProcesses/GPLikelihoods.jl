"""
    LatentGP(f<:GP, x<:AbstractVector, Φ)

 - `fx` is a `FiniteGP`.
 - `Φ` is the log likelihood function which maps input of type `Tx` to `Real`.
    
"""
struct LatentGP{T<:AbstractGPs.FiniteGP}
    fx::T
    Φ
end

function Distributions.rand(rng::AbstractRNG, lgp::LatentGP)
    f = rand(rng, lgp.fx)
    y = rand(rng, lgp.Φ(f))
    return (f=f, y=y)
end

"""
    logpdf(lgp::LatentGP, y::NamedTuple{(:f, :y)})

```math
    log p(y, f; x)
```
Returns the joint log density of the gaussian process output `f` and real output `y`.
"""
function Distributions.logpdf(lgp::LatentGP, y::NamedTuple{(:f, :y)})
    return logpdf(lgp.fx, y.f) + logpdf(lgp.Φ(y.f), y.y)
end

function Distributions.logpdf(lgp::LatentGP, y, f)
    return logpdf(lgp.fx, f) + logpdf(lgp.Φ(f), y)
end
