"""
    GaussianLikelihood(σ²)

Gaussian likelihood with `σ²` variance. This is to be used if we assume that the
uncertainity associated with the data follows a Gaussian distribution.

```math
    p(y|f) = Normal(y | f, σ²)
```
On calling, this would return a normal distribution with mean `f` and variance σ².
"""
struct GaussianLikelihood{T<:Real}
    σ²::Vector{T}
end

GaussianLikelihood() = GaussianLikelihood(1e-6)

GaussianLikelihood(σ²::Real) = GaussianLikelihood([σ²])

@functor GaussianLikelihood

(l::GaussianLikelihood)(f::Real) = Normal(f, sqrt(first(l.σ²)))

(l::GaussianLikelihood)(fs::AbstractVector{<:Real}) = MvNormal(fs, sqrt(first(l.σ²)))

"""
    HeteroscedasticGaussianLikelihood(l::AbstractLink=ExpLink())

Heteroscedastic Gaussian likelihood. 
This is a Gaussian likelihood whose mean and the log of whose variance are functions of the
latent process.

```math
    p(y|[f, g]) = Normal(y | f, l(g))
```
On calling, this would return a normal distribution with mean `f` and variance `l(g)`.
Where `l` is link going from R to R^+
"""
struct HeteroscedasticGaussianLikelihood{Tl<:AbstractLink}
    invlink::Tl
end

HeteroscedasticGaussianLikelihood() = HeteroscedasticGaussianLikelihood(ExpLink())

(l::HeteroscedasticGaussianLikelihood)(f::AbstractVector{<:Real}) = Normal(f[1], l.invlink(f[2]))

(l::HeteroscedasticGaussianLikelihood)(fs::AbstractVector) = MvNormal(first.(fs), l.invlink.(last.(fs)))
