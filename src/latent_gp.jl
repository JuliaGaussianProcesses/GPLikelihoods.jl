"""
    LatentGP(f<:GP, x<:AbstractVector, Φ)

 - `f` is a Gaussian Process.
 - `x` is an `AbstractVector` of inputs of type `Tx`.
 - `Φ` is the log likelihood function which maps input of type `Tx` to `Real`.
    
"""
struct LatentGP{T<:GP,Tx}
    f::T
    x::AbstractVector{Tx}
    σ²
    Φ
    function LatentGP(f::GP, x::AbstractVector{Tx}, σ², Φ) where Tx
        return new{typeof(f),Tx}(f, x, σ², Φ)
    end
end

function LatentGP(f::GP, x::AbstractVector{Tx}, Φ) where Tx
        return LatentGP(f, x, 0.1, Φ)
    end


function mean(fx::LatentGP)
    return AbstractGPs.mean(fx.f, fx.x)
end

function cov(fx::LatentGP)
    return AbstractGPs.cov(fx.f, fx.x)
end
