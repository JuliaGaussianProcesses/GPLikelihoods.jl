"""
    PoissonLikelihood(invlink=ExpLink())

Poisson likelihood with rate as exponential of samples from GP `f`. This is to be used if
we assume that the uncertainity associated with the data follows a Poisson distribution.
"""
struct PoissonLikelihood{L<:AbstractLink}
    invlink::L
end

PoissonLikelihood() = PoissonLikelihood(ExpLink())

@functor PoissonLikelihood

(l::PoissonLikelihood)(f::Real) = Poisson(l.invlink(f))

(l::PoissonLikelihood)(fs::AbstractVector{<:Real}) = Product(Poisson.(l.invlink.(fs)))
