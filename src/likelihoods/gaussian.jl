"""
    GaussianLikelihood(σ²)

Gaussian likelihood with `σ²` variance. This is to be used if we assume that the
uncertainity associated with the data follows a Gaussian distribution.

```math
    p(y|f) = \\operatorname{Normal}(y | f, σ²)
```
On calling, this would return a normal distribution with mean `f` and variance σ².
"""
struct GaussianLikelihood{T<:Real} <: AbstractLikelihood
    σ²::Vector{T}
end

GaussianLikelihood() = GaussianLikelihood(1e-6)

GaussianLikelihood(σ²::Real) = GaussianLikelihood([σ²])

@functor GaussianLikelihood

(l::GaussianLikelihood)(f::Real) = Normal(f, sqrt(first(l.σ²)))

(l::GaussianLikelihood)(fs::AbstractVector{<:Real}) = MvNormal(fs, first(l.σ²) * I)

function expected_loglikelihood(
    ::AnalyticExpectation,
    lik::GaussianLikelihood,
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    return sum(_gaussian_exp_loglikelihood_kernel.(lik.σ², q_f, y))
end

function expected_loglikelihood(
    ::AnalyticExpectation,
    liks::AbstractVector{<:GaussianLikelihood},
    q_f::AbstractVector{<:Normal},
    y::AbstractVector{<:Real},
)
    return sum(_gaussian_exp_loglikelihood_kernel.(getfield.(liks, :σ²), q_f, y))
end

function _gaussian_exp_loglikelihood_kernel(σ², q_f, y)
    return -0.5 * (log(2π) + log(σ²) + ((y - mean(q_f))^2 + var(q_f)) / σ²)
end

default_expectation_method(::GaussianLikelihood) = AnalyticExpectation()

"""
    HeteroscedasticGaussianLikelihood(l=exp)

Heteroscedastic Gaussian likelihood. 
This is a Gaussian likelihood whose mean and variance are functions of
latent processes.

```math
    p(y|[f, g]) = \\operatorname{Normal}(y | f, sqrt(l(g)))
```
On calling, this would return a normal distribution with mean `f` and variance `l(g)`.
Where `l` is link going from R to R^+
"""
struct HeteroscedasticGaussianLikelihood{Tl<:AbstractLink} <: AbstractLikelihood
    invlink::Tl
end

HeteroscedasticGaussianLikelihood(l=exp) = HeteroscedasticGaussianLikelihood(link(l))

function (l::HeteroscedasticGaussianLikelihood)(f::AbstractVector{<:Real})
    return Normal(f[1], sqrt(l.invlink(f[2])))
end

function (l::HeteroscedasticGaussianLikelihood)(fs::AbstractVector)
    return MvNormal(first.(fs), Diagonal(l.invlink.(last.(fs))))
end
