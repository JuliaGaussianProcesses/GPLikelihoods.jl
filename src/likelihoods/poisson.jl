"""
    PoissonLikelihood()

Poisson likelihood with rate as exponential of samples from GP `f`. This is to be used if
we assume that the uncertainity associated with the data follows a Poisson distribution.
"""
struct PoissonLikelihood <: Likelihood end

(l::PoissonLikelihood)(f::Real) = Poisson(exp(f))

(l::PoissonLikelihood)(fs::AbstractVector{<:Real}) = Product(Poisson.(exp.(fs)))
