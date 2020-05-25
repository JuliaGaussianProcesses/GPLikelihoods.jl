"""
    LatentGP(f<:GP, x<:AbstractVector, Φ)

 - `fx` is a `FiniteGP`.
 - `Φ` is the log likelihood function which maps input of type `Tx` to `Real`.
    
"""
struct LatentGP{T<:AbstractGPs.FiniteGP}
    fx::T
    Φ
    function LatentGP(fx::AbstractGPs.FiniteGP, Φ) where Tx
        return new{typeof(fx)}(fx, Φ)
    end    
end

function LatentGP(f::GP, x::AbstractVector{Tx}, σ², Φ) where Tx
        return LatentGP(f(x, σ²), Φ)
end

function LatentGP(f::GP, x::AbstractVector{Tx}, Φ) where Tx
        return LatentGP(f(x), Φ)
end

function Distributions.rand(rng::AbstractRNG, lgp::LatentGP)
    v = rand(rng, lgp.fx)
    y = rand(rng, lgp.Φ(v))
    return (v=v, y=y)
end

function Distributions.logpdf(lgp::LatentGP, y::NamedTuple{(:v, :y)})
    return logpdf(lgp.fx, y.v) + logpdf(lgp.Φ(y.v), y.y)
end